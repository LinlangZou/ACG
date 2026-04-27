#pragma once

#include "Core.h"
#include "Imaging.h"
#include "Sampling.h"
#include <algorithm>

#pragma warning( disable : 4244)
#pragma warning( disable : 4305) // Double to float

constexpr float kMaterialEpsilon = 1e-6f;

class BSDF;

class ShadingData
{
public:
	Vec3 x;
	Vec3 wo;
	Vec3 sNormal;
	Vec3 gNormal;
	float tu;
	float tv;
	Frame frame;
	BSDF* bsdf;
	float t;
	ShadingData() {}
	ShadingData(Vec3 _x, Vec3 n)
	{
		x = _x;
		gNormal = n;
		sNormal = n;
		bsdf = NULL;
	}
};

class ShadingHelper
{
public:
	static float fresnelDielectric(float cosTheta, float iorInt, float iorExt)
	{
		if (cosTheta < 0.0f) {
			std::swap(iorExt, iorInt);
			cosTheta = fabsf(cosTheta);
		}
		float eta = iorInt / iorExt;
		float sinThetaT2 = eta * eta * (1 - cosTheta * cosTheta);

		if (sinThetaT2 > 1.0f) return 1.0f;

		float cosThetaT = sqrt(1.0f - sinThetaT2);
		float Rs = (cosTheta - eta * cosThetaT) / (cosTheta + eta * cosThetaT);
		float Rp = (eta * cosTheta - cosThetaT) / (eta * cosTheta + cosThetaT);
		return 0.5f * (Rs * Rs + Rp * Rp);
	}
	static Colour fresnelConductor(float cosTheta, Colour ior, Colour k)
	{
		Colour eta2 = ior * ior;
		Colour k2 = k * k;
		float cosTheta2 = cosTheta * cosTheta;
		float sinTheta2 = 1.0f - cosTheta2;
		Colour twoEtaCosTheta = ior * (2.0f * cosTheta);
		Colour cosTheta2C(cosTheta2, cosTheta2, cosTheta2);
		Colour sinTheta2C(sinTheta2, sinTheta2, sinTheta2);

		Colour Rs = (eta2 + k2 - twoEtaCosTheta + cosTheta2C) / (eta2 + k2 + twoEtaCosTheta + cosTheta2C);
		Colour Rp = ((eta2 + k2) * cosTheta2 - twoEtaCosTheta + sinTheta2C) / ((eta2 + k2) * cosTheta2 + twoEtaCosTheta + sinTheta2C);

		return (Rs * Rs + Rp * Rp) * 0.5f;
	}
	static float lambdaGGX(Vec3 wi, float alpha)
	{
		float cosTheta = fabsf(wi.z);
		if (cosTheta < kMaterialEpsilon) return 0.0f;

		float sinTheta2 = 1.0f - cosTheta * cosTheta;
		float tanTheta2 = sinTheta2 / (cosTheta * cosTheta);

		float alpha2 = alpha * alpha;
		return (sqrtf(1.0f + alpha2 * tanTheta2) - 1.0f) * 0.5f;
	}
	static float Gggx(Vec3 wi, Vec3 wo, float alpha)
	{
		float lambdaI = lambdaGGX(wi, alpha);
		float lambdaO = lambdaGGX(wo, alpha);
		float G1 = 1 / (1 + lambdaI);
		float G2 = 1 / (1 + lambdaO);

		return G1 * G2;
	}
	static float Dggx(Vec3 h, float alpha)
	{
		float alpha2 = alpha * alpha;
		float cosTheta = fabs(h.z);
		float cosTheta2 = cosTheta * cosTheta;

		float denominator = (cosTheta2 * (alpha2 - 1.0f) + 1.0f);
		denominator = M_PI * denominator * denominator;

		return alpha2 / denominator;
	}
	static Vec3 sampleGGX(Vec3 wo, float alpha, Sampler* sampler)
	{
		float r1 = sampler->next();
		float r2 = sampler->next();

		float phi = 2.0f * M_PI * r1;
		float theta = acosf(sqrtf((1 - r2) / (r2 * (alpha * alpha - 1) + 1)));

		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}
};

class BSDF
{
public:
	Colour emission;
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isPureSpecular() = 0;
	virtual bool isTwoSided() = 0;
	bool isLight()
	{
		return emission.Lum() > 0 ? true : false;
	}
	void addLight(Colour _emission)
	{
		emission = _emission;
	}
	Colour emit(const ShadingData& shadingData, const Vec3& wi)
	{
		return emission;
	}
	virtual float mask(const ShadingData& shadingData) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) const {
		return 1.0f / (2.0f * M_PI);
	}
};

class DiffuseBSDF : public BSDF
{
public:
	Texture* albedo;
	DiffuseBSDF() = default;
	DiffuseBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class MirrorBSDF : public BSDF
{
public:
	Texture* albedo;
	MirrorBSDF() = default;
	MirrorBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	float SchlickFresnel(float cosTheta, float ior)
	{
		float R0 = pow((ior - 1.f) / (ior + 1.f), 2.0f);
		return R0 + (1.0f - R0) * pow(1.0f - cosTheta, 5.0f);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		pdf = 1.f;
		float cosTheta = fabsf(woLocal.z);
		float ior = 0.f;
		float F = SchlickFresnel(cosTheta, ior);
		Vec3 wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * F;
		return shadingData.frame.toWorld(wiLocal);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return Colour(0.f, 0.f, 0.f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 0.f;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class ConductorBSDF : public BSDF
{
public:
	Texture* albedo;
	Colour eta;
	Colour k;
	float alpha;
	ConductorBSDF() = default;
	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness)
	{
		albedo = _albedo;
		eta = _eta;
		k = _k;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{


		Vec3 wo = shadingData.frame.toLocal(shadingData.wo);
		Vec3 h = ShadingHelper::sampleGGX(wo, alpha, sampler);
		Vec3 wi = -wo + h * 2.0f * (wo.dot(h));

		if (wi.z <= 0.0f) {
			pdf = 0.f;
			reflectedColour = Colour(0.f, 0.f, 0.f);
			return Vec3(0.0f, 0.f, 0.f);
		} 

		float cosTheta = fabsf(wo.z);
		Colour F = ShadingHelper::fresnelConductor(cosTheta, eta, k);
		float D = ShadingHelper::Dggx(h, alpha);
		float G = ShadingHelper::Gggx(wi, wo, alpha);
		if (D <= kMaterialEpsilon) D = kMaterialEpsilon;
		

		float denominator = 4.0f * std::max(1e-6f, wo.z) * std::max(1e-6f, wi.z);
		float MF = G * D / denominator;
		
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * F * MF;

		float dotHO = std::max(1e-6f, Dot(wo, h)); 
		float dotNH = std::max(1e-6f, h.z);
		pdf = (D * dotNH) / (4.0f * dotHO);

		return shadingData.frame.toWorld(wi);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		

		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f) return Colour(0.f, 0.f, 0.f);
		Vec3 h = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(h, alpha);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		Colour F = ShadingHelper::fresnelConductor(fabsf(woLocal.dot(h)), eta, k);
		float MF = G * D / (4 * fabsf(woLocal.z) * fabsf(wi.z));
		return albedo->sample(shadingData.tu, shadingData.tv) * F * MF;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f) return 0.0f;

		Vec3 h = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(h, alpha);

		return D * fabsf(h.z) / (4.0f * Dot(woLocal, h));
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class GlassBSDF : public BSDF {
public:
    Texture* albedo;
    float iorInt;
    float iorExt;

    GlassBSDF(Texture* _albedo, float _iorInt = 1.5f, float _iorExt = 1.0f)
        : albedo(_albedo), iorInt(_iorInt), iorExt(_iorExt) {}

    Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) override {
        Vec3 wo = shadingData.wo;
        Vec3 normal = shadingData.sNormal;
        Vec3 woLocal = shadingData.frame.toLocal(wo);
        float cosThetaI = woLocal.z;
        bool entering = cosThetaI > 0;
                float etaI = entering ? iorExt : iorInt;
        float etaT = entering ? iorInt : iorExt;
        float eta = etaI / etaT;
        float F = ShadingHelper::fresnelDielectric(fabsf(cosThetaI), etaI, etaT);
		
        if (sampler->next() < F) {
            Vec3 wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z); 
            reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * F;
            pdf = F;
            return shadingData.frame.toWorld(wiLocal);
        } 
        else { 
			float cosThetaI = fabsf(woLocal.z);
			float sinThetaI = sqrtf(1.0f - cosThetaI * cosThetaI);
			float sinThetaT = eta * sinThetaI;
			float cosThetaT = sqrtf(1.0f - sinThetaT * sinThetaT);
			float phi = atan2f(woLocal.y, woLocal.x) + M_PI;
			Vec3 wiLocal;
			wiLocal.x = sinThetaT * cosf(phi);
			wiLocal.y = sinThetaT * sinf(phi);
			wiLocal.z = entering ? -cosThetaT : cosThetaT;
			float transmitScale = (etaT * etaT) / (etaI * etaI);
			reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * (1 - F) * transmitScale;
			pdf = 1 - F;
			return shadingData.frame.toWorld(wiLocal.normalize());
        }
    }

    Colour evaluate(const ShadingData& shadingData, const Vec3& wi) override {
		return Colour(0.f,0.f,0.f);
    }

    float PDF(const ShadingData& shadingData, const Vec3& wi) override {
        return 0.0f;
    }

    bool isPureSpecular() override { return true; }
    bool isTwoSided() override { return false; }

    float mask(const ShadingData& shadingData) override {
        return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
    }
};
class DielectricBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	DielectricBSDF() = default;
	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Dielectric sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class OrenNayarBSDF : public BSDF
{
public:
	Texture* albedo;
	float sigma;
	OrenNayarBSDF() = default;
	OrenNayarBSDF(Texture* _albedo, float _sigma)
	{
		albedo = _albedo;
		sigma = _sigma;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with OrenNayar sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class PlasticBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	PlasticBSDF() = default;
	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	float alphaToPhongExponent()
	{
		return (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Plastic sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class LayeredBSDF : public BSDF
{
public:
	BSDF* base;
	Colour sigmaa;
	float thickness;
	float intIOR;
	float extIOR;
	LayeredBSDF() = default;
	LayeredBSDF(BSDF* _base, Colour _sigmaa, float _thickness, float _intIOR, float _extIOR)
	{
		base = _base;
		sigmaa = _sigmaa;
		thickness = _thickness;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add code to include layered sampling
		return base->sample(shadingData, sampler, reflectedColour, pdf);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code for evaluation of layer
		return base->evaluate(shadingData, wi);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code to include PDF for sampling layered BSDF
		return base->PDF(shadingData, wi);
	}
	bool isPureSpecular()
	{
		return base->isPureSpecular();
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return base->mask(shadingData);
	}
};