#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include <thread>
#include <functional>
#include <mutex>
#include <queue>
#include <OpenImageDenoise/oidn.hpp>
#include <algorithm>

#ifndef RT_ENABLE_MULTITHREADING
#define RT_ENABLE_MULTITHREADING 1
#endif

class VPL {
public:
	ShadingData shadingData;
	Colour Ls;
	Vec3 position() const { return shadingData.x; }
	void init(ShadingData _shadingData, Colour c) {
		shadingData = _shadingData;
		Ls = c;
	}
};

class RayTracer
{
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom *samplers;
	std::thread **threads;
	int numProcs;
	std::vector<VPL> vpls;

	struct BlockStats {
		std::vector<Colour> pixelSamples;
		Colour sum;
		Colour sumSquares;
		float variance = 0.0f;
		int allocatedSamples = 4;
		int x0, y0, x1, y1;
	};

	int blockSize = 32;
	int numBlocksX = 0;         
	int numBlocksY = 0;    
	std::vector<BlockStats> blocks;
	int totalSamples = 256;

	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new MitchellNetravaliFilter());
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = std::max(1, (int)sysInfo.dwNumberOfProcessors);
		threads = new std::thread*[numProcs];
		samplers = new MTRandom[numProcs];
		clear();

		for (int i = 0; i < numProcs; i++) {
			samplers[i] = MTRandom(i + 1);
		}

		blockSize = 32;
		numBlocksX = (scene->camera.width + blockSize - 1) / blockSize;
		numBlocksY = (scene->camera.height + blockSize - 1) / blockSize;
		blocks.resize(numBlocksX * numBlocksY);

		for (int by = 0; by < numBlocksY; ++by) {
			for (int bx = 0; bx < numBlocksX; ++bx) {
				auto& block = blocks[by * numBlocksX + bx];
				block.x0 = bx * blockSize;
				block.y0 = by * blockSize;
				block.x1 = std::min(block.x0 + blockSize, static_cast<int>(scene->camera.width));
				block.y1 = std::min(block.y0 + blockSize, static_cast<int>(scene->camera.height));
				block.pixelSamples.resize((block.x1 - block.x0) * (block.y1 - block.y0), Colour(0, 0, 0));
			}
		}
	}
	void clear()
	{
		film->clear();
		for (auto& block : blocks) {
			block.sum = Colour(0, 0, 0);
			block.sumSquares = Colour(0, 0, 0);
			block.variance = 0.0f;
			block.allocatedSamples = 4;
			std::fill(block.pixelSamples.begin(), block.pixelSamples.end(), Colour(0, 0, 0));
		}
	}

#define MAX_DEPTH 8

	void traceVPLs(Sampler* sampler, int N_VPLs) {
		vpls.clear();

		for (int i = 0; i < N_VPLs; ++i) {
			float lightPmf;
			Light* light = scene->sampleLight(sampler, lightPmf);
			if (!light || lightPmf <= 1e-6f) continue;
			float pdfPosition, pdfDirection;
			Vec3 lightPos = light->samplePositionFromLight(sampler, pdfPosition);
			Vec3 lightDir = light->sampleDirectionFromLight(sampler, pdfDirection);
			Colour Le = light->evaluate( -lightDir);
			Colour throughput = Le * std::abs(Dot(lightDir, light->normal(ShadingData(), lightDir)))
				/ (lightPmf * pdfPosition * pdfDirection);
			Ray ray(lightPos + lightDir * EPSILON, lightDir);
			VPLTracePath(ray, throughput, sampler, 0);
		}
	}

	void VPLTracePath(Ray& r, Colour pathThroughput, Sampler* sampler, int depth) {
		if (depth >= 5) return; 
		
		IntersectionData isect = scene->traverse(r);
		if (isect.t >= FLT_MAX) return;

		ShadingData sd = scene->calculateShadingData(isect, r);


		if (!sd.bsdf->isPureSpecular()) {
			VPL newVPL;
			newVPL.shadingData = sd;
			newVPL.Ls = pathThroughput;
			vpls.push_back(newVPL);
		}

		float rrProb = std::min(pathThroughput.Lum(), 0.95f);
		if (sampler->next() > rrProb) return;
		pathThroughput = pathThroughput / rrProb;


		Colour bsdfVal;
		float pdf;
		Vec3 wi = sd.bsdf->sample(sd, sampler, bsdfVal, pdf);
		if (pdf < 1e-6f) return;

		pathThroughput = pathThroughput * bsdfVal * std::abs(Dot(wi, sd.sNormal)) / pdf;

		Ray nextRay(sd.x + wi * EPSILON, wi);
		VPLTracePath(nextRay, pathThroughput, sampler, depth + 1);
	}

	Colour computeDirect(ShadingData shadingData, Sampler* sampler)
	{
		Colour direct;
		Colour result;
		if (shadingData.bsdf->isPureSpecular() == true)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		float pdf;
		Colour emitted;
		Vec3 p = light->sample(shadingData, sampler, emitted, pdf);
		if (light->isArea())
		{

			Vec3 wi = p - shadingData.x;
			float l = wi.lengthSq();
			wi = wi.normalize();
			float GTerm = (std::max(Dot(wi, shadingData.sNormal), 0.0f) * std::max(-Dot(wi, light->normal(shadingData, wi)), 0.0f)) / l;
			if (GTerm > 0)
			{
		
				if (scene->visible(shadingData.x, p))
				{
				
					direct = shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		else
		{
		
			Vec3 wi = p;
			float GTerm = std::max(Dot(wi, shadingData.sNormal), 0.0f);
			if (GTerm > 0)
			{
			
				if (scene->visible(shadingData.x, shadingData.x + (p * 10000.0f)))
				{
				
					direct = shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		for (const VPL& vpl : vpls) {
			Vec3 x_i = vpl.position();
			Vec3 wi = (x_i - shadingData.x).normalize();

		
			float dist2 = (x_i - shadingData.x).lengthSq();
			float cosTheta = std::abs(Dot(wi, shadingData.sNormal));
			float cosThetaVPL = std::abs(Dot(-wi, vpl.shadingData.sNormal));
			float G = cosTheta * cosThetaVPL / dist2;
			if (G < 1e-6f) continue;
			if (scene->visible(shadingData.x, x_i)) {
				Colour frCamera = shadingData.bsdf->evaluate(shadingData, wi);
				Colour frVPL = vpl.shadingData.bsdf->evaluate(vpl.shadingData, -wi);
				result = result + frCamera * frVPL * G * vpl.Ls;
			}
		}
		return  direct + result;
	}

	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler, bool canHitLight = true)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				if (canHitLight == true)
				{
					return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
				}
				else
				{
					return Colour(0.0f, 0.0f, 0.0f);
				}
			}
			Colour direct = pathThroughput * computeDirect(shadingData, sampler);
			if (depth > MAX_DEPTH)
			{
				return direct;
			}
			float russianRouletteProbability = std::min(pathThroughput.Lum(), 0.9f);
			if (sampler->next() < russianRouletteProbability)
			{
				pathThroughput = pathThroughput / russianRouletteProbability;
			}
			else
			{
				return direct;
			}
			Colour bsdf;
			float pdf;
			Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, bsdf, pdf);

			pathThroughput = pathThroughput * bsdf * fabsf(Dot(wi, shadingData.sNormal)) / pdf;
			r.init(shadingData.x + (wi * EPSILON), wi);
			
			return (direct + pathTrace(r, pathThroughput, depth + 1, sampler, shadingData.bsdf->isPureSpecular()));
		}
		return scene->background->evaluate( r.dir);
	}
	Colour direct(Ray& r, Sampler* sampler)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return computeDirect(shadingData, sampler);
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}	

	Colour albedo(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return shadingData.bsdf->evaluate(shadingData, Vec3(0, 1, 0));
		}
		return scene->background->evaluate( r.dir);
	}
	Colour viewNormals(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX)
		{
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	void computeVarianceAndAllocate() {
		float totalVariance = 0.0f;

		for (auto& block : blocks) {
			int numPixels = (block.x1 - block.x0) * (block.y1 - block.y0);
			if (numPixels == 0) continue;

			Colour mean = block.sum / numPixels;
			Colour meanSquares = block.sumSquares / numPixels;
			block.variance = (meanSquares - mean * mean).Lum();
			totalVariance += block.variance;
		}

		const int remainingSamples = totalSamples - film->SPP;
		for (auto& block : blocks) {
			float weight = totalVariance > 0 ? block.variance / totalVariance : 1.0f / blocks.size();
			block.allocatedSamples = std::max(1, (int)std::round(weight * remainingSamples));
		}
	}

	void render()
	{
		traceVPLs(&samplers[0], 32);
		static const int TILE_SIZE = 32;
		film->incrementSPP();

		int numTilesX = (film->width + TILE_SIZE - 1) / TILE_SIZE;
		int numTilesY = (film->height + TILE_SIZE - 1) / TILE_SIZE;
		std::vector<float> hdrpixels(film->width * film->height * 3, 0.0f);

		auto renderTile = [&](int tileX, int tileY, int threadId)
			{
				int startX = tileX * TILE_SIZE;
				int startY = tileY * TILE_SIZE;
				int endX = std::min(startX + TILE_SIZE, (int)film->width);
				int endY = std::min(startY + TILE_SIZE, (int)film->height);

				Sampler* localSampler = &samplers[threadId];

				for (int y = startY; y < endY; y++)
				{
					for (int x = startX; x < endX; x++)
					{
						float px = x + 0.5f;
						float py = y + 0.5f;

						Ray ray = scene->camera.generateRay(px, py);

						Colour pathThroughput(1.0f, 1.0f, 1.0f);
						Colour col = pathTrace(ray, pathThroughput, 5, localSampler);
						//Colour col = direct(ray, localSampler);
						//Colour col = viewNormals(ray);
						film->splat(px, py, col);

						int globalIndex = (y * film->width + x) * 3;
						hdrpixels[globalIndex + 0] = col.r;
						hdrpixels[globalIndex + 1] = col.g;
						hdrpixels[globalIndex + 2] = col.b;
					}
				}
			};

#if RT_ENABLE_MULTITHREADING
		int numThreads = numProcs;
		std::vector<std::thread> workers;
		workers.reserve(numThreads);
		auto workerFunc = [&](int threadId)
			{
				for (int tileY = 0; tileY < numTilesY; tileY++)
				{
					for (int tileX = 0; tileX < numTilesX; tileX++)
					{
						if (((tileY * numTilesX) + tileX) % numThreads == threadId)
						{
							renderTile(tileX, tileY, threadId);
						}
					}
				}
			};
		for (int i = 0; i < numThreads; i++)
		{
			workers.emplace_back(workerFunc, i);
		}
		for (auto& w : workers) {
			w.join();
		}
#else
		for (int tileY = 0; tileY < numTilesY; tileY++)
		{
			for (int tileX = 0; tileX < numTilesX; tileX++)
			{
				renderTile(tileX, tileY, 0);
			}
		}
#endif

		// Denoise
		oidn::DeviceRef device = oidn::newDevice();
		device.commit();
		oidn::BufferRef buffer = device.newBuffer(film->width * film->height * 3 * sizeof(float));
		std::memcpy(buffer.getData(), hdrpixels.data(), buffer.getSize());
		oidn::FilterRef filter = device.newFilter("RT");
		filter.setImage("color", buffer, oidn::Format::Float3, film->width, film->height);
		filter.setImage("output", buffer, oidn::Format::Float3, film->width, film->height);
		filter.set("hdr", true);
		filter.commit();
		filter.execute();
		// Store the output
		std::memcpy(hdrpixels.data(), buffer.getData(), buffer.getSize());

		// Draw final frame
		for (int y = 0; y < film->height; y++)
		{
			for (int x = 0; x < film->width; x++)
			{
				int index = (y * film->width + x) * 3;
				float r_value = hdrpixels[index + 0] * 255.0f;
				float g_value = hdrpixels[index + 1] * 255.0f;
				float b_value = hdrpixels[index + 2] * 255.0f;

				r_value = std::clamp(r_value, 0.0f, 255.0f);
				g_value = std::clamp(g_value, 0.0f, 255.0f);
				b_value = std::clamp(b_value, 0.0f, 255.0f);

				unsigned char r = static_cast<unsigned char>(r_value);
				unsigned char g = static_cast<unsigned char>(g_value);
				unsigned char b = static_cast<unsigned char>(b_value);

				film->tonemap(x, y, r, g, b);
				canvas->draw(x, y, r, g, b);
			}
		}
	}



	int getSPP()
	{
		return film->SPP;
	}
	void saveHDR(std::string filename)
	{
		film->save(filename);
	}
	void savePNG(std::string filename)
	{
		stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3);
	}
};