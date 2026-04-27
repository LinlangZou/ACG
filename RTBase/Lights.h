#pragma once

#include "Core.h"
#include "Geometry.h"
#include "Materials.h"
#include "Sampling.h"

#pragma warning( disable : 4244)

class SceneBounds
{
public:
	Vec3 sceneCentre;
	float sceneRadius;
};

class Light
{
public:
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf) = 0;
	virtual Colour evaluate(const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isArea() = 0;
	virtual Vec3 normal(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float totalIntegratedPower() = 0;
	virtual Vec3 samplePositionFromLight(Sampler* sampler, float& pdf) = 0;
	virtual Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf) = 0;
};

class AreaLight : public Light
{
public:
	Triangle* triangle = NULL;
	Colour emission;
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		emittedColour = emission;
		return triangle->sample(sampler, pdf);
	}
	Colour evaluate(const Vec3& wi)
	{
		if (Dot(wi, triangle->gNormal()) < 0)
		{
			return emission;
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 1.0f / triangle->area;
	}
	bool isArea()
	{
		return true;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return triangle->gNormal();
	}
	float totalIntegratedPower()
	{
		return (triangle->area * emission.Lum());
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		return triangle->sample(sampler, pdf);
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		// Add code to sample a direction from the light
		Vec3 wi = Vec3(0, 0, 1);
		pdf = 1.0f;
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wi);
	}
};

class BackgroundColour : public Light
{
public:
	Colour emission;
	BackgroundColour(Colour _emission)
	{
		emission = _emission;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		reflectedColour = emission;
		return wi;
	}
	Colour evaluate(const Vec3& wi)
	{
		return emission;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return SamplingDistributions::uniformSpherePDF(wi);
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		return emission.Lum() * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 4 * M_PI * use<SceneBounds>().sceneRadius * use<SceneBounds>().sceneRadius;
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};

class EnvironmentMap : public Light
{
public:
    struct Distribution1D {
        std::vector<float> cdf;
        float funcInt;
        int count() const { return cdf.empty() ? 0 : cdf.size() - 1; }

        void build(const std::vector<float>& func) {
            int n = func.size();
            cdf.resize(n + 1);
            cdf[0] = 0.0f;
            for (int i = 1; i <= n; ++i)
                cdf[i] = cdf[i - 1] + func[i - 1];
            funcInt = cdf[n];
            if (funcInt == 0) {
                for (int i = 1; i <= n; ++i)
                    cdf[i] = float(i) / float(n);
            }
            else {
                for (int i = 1; i <= n; ++i)
                    cdf[i] /= funcInt;
            }
        }
        float sample(float u, float* pdf) const {
            int offset = std::distance(cdf.begin(), std::lower_bound(cdf.begin(), cdf.end(), u)) - 1;
            offset = std::max(0, std::min(offset, count() - 1));
            if (pdf) *pdf = (funcInt > 0) ? (cdf[offset + 1] - cdf[offset]) * count() : 0;
            float du = u - cdf[offset];
            if ((cdf[offset + 1] - cdf[offset]) > 0)
                du /= (cdf[offset + 1] - cdf[offset]);
            return (offset + du) / count();
        }
    };

    Distribution1D marginal;
    std::vector<Distribution1D> conditionals;
    Texture* env;

    EnvironmentMap(Texture* _env) : env(_env) {
        int width = env->width;
        int height = env->height;
        std::vector<float> marginalFunc(height);

        conditionals.resize(height);
        for (int y = 0; y < height; ++y) {
            std::vector<float> func(width);
            float theta = M_PI * (y + 0.5f) / height;
            float sinTheta = std::sin(theta);
            for (int x = 0; x < width; ++x) {
                Colour texel = env->texels[y * width + x];
                func[x] = texel.Lum() * sinTheta;
            }
            conditionals[y].build(func);
            marginalFunc[y] = conditionals[y].funcInt; 
        }

        marginal.build(marginalFunc);
    }

    Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		Vec3 wi = sampleDirectionFromLight(sampler, pdf);
		reflectedColour = evaluate(wi);
		return wi;
    }

    Colour evaluate(const Vec3& wi) {
        float phi = std::atan2(wi.z, wi.x);
        phi = (phi < 0) ? phi + 2 * M_PI : phi;
        float u = phi / (2 * M_PI);
        float theta = std::acos(wi.y);
        float v = theta / M_PI;
        return env->sample(u, v);
    }

    float PDF(const ShadingData& shadingData, const Vec3& wi) {
        float phi = std::atan2(wi.z, wi.x);
        if (phi < 0) phi += 2 * M_PI;
        float u = phi / (2 * M_PI);
        float theta = std::acos(wi.y);
        float v = theta / M_PI;

        int y = std::max(0, std::min(int(v * env->height), env->height - 1));
        int x = std::max(0, std::min(int(u * env->width), env->width - 1));

        float pdfMarginal = (marginal.cdf[y + 1] - marginal.cdf[y]) * env->height;
        float pdfConditional = (conditionals[y].cdf[x + 1] - conditionals[y].cdf[x]) * env->width;
        float uvPdf = pdfMarginal * pdfConditional;

        float sinTheta = std::sin(theta);
        if (sinTheta <= 0) return 0;

        return uvPdf / (2 * M_PI * M_PI * sinTheta);
    }
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		float total = 0;
		for (int i = 0; i < env->height; i++)
		{
			float st = sinf(((float)i / (float)env->height) * M_PI);
			for (int n = 0; n < env->width; n++)
			{
				total += (env->texels[(i * env->width) + n].Lum() * st);
			}
		}
		total = total / (float)(env->width * env->height);
		return total * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 1.0f / (4 * M_PI * SQ(use<SceneBounds>().sceneRadius));
		return p;
	}
    Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf) {
        float u1 = sampler->next();
        float u2 = sampler->next();
        float pdfMarginal;
        int y = std::distance(marginal.cdf.begin(), std::lower_bound(marginal.cdf.begin(), marginal.cdf.end(), u1)) - 1;
        y = std::max(0, std::min(y, env->height - 1));
        pdfMarginal = (marginal.cdf[y + 1] - marginal.cdf[y]) * env->height;
        float pdfConditional;
        float u = conditionals[y].sample(u2, &pdfConditional);
        float phi = 2 * M_PI * u;
        float theta = M_PI * (y + 0.5f) / env->height;
        float sinTheta = std::sin(theta);
        Vec3 wi(
            std::cos(phi) * sinTheta,
            std::cos(theta),
            std::sin(phi) * sinTheta
        );
        pdf = (pdfMarginal * pdfConditional) / (2 * M_PI * M_PI * sinTheta);
        return wi;
    }
};
