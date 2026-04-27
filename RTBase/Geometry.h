#pragma once

#include "Core.h"
#include "Sampling.h"
#define cmax(a,b)            (((a) > (b)) ? (a) : (b))
#define cmin(a,b)            (((a) < (b)) ? (a) : (b))

class Ray
{
public:
	Vec3 o;
	Vec3 dir;
	Vec3 invDir;
	Ray()
	{
	}
	Ray(Vec3 _o, Vec3 _d)
	{
		init(_o, _d);
	}
	void init(Vec3 _o, Vec3 _d)
	{
		o = _o;
		dir = _d;
		invDir = Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
	}
	Vec3 at(const float t) const
	{
		return (o + (dir * t));
	}
};

class Plane
{
public:
	Vec3 n;
	float d;
	void init(Vec3& _n, float _d)
	{
		n = _n;
		d = _d;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		float denom = Dot(n, r.dir);
		if (denom == 0) { return false; }

		t = (d - Dot(n, r.o)) / denom;
		return (t >= 0);
	}
};

#define EPSILON 0.0000001f

class Triangle
{
public:
	Vertex vertices[3];
	Vec3 e1; // Edge 1
	Vec3 e2; // Edge 2
	Vec3 n; // Geometric Normal
	float area; // Triangle area
	float d; // For ray triangle if needed
	unsigned int materialIndex;
	void init(Vertex v0, Vertex v1, Vertex v2, unsigned int _materialIndex)
	{
		materialIndex = _materialIndex;
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
		e1 = vertices[2].p - vertices[1].p;
		e2 = vertices[0].p - vertices[2].p;
		n = e1.cross(e2).normalize();
		area = e1.cross(e2).length() * 0.5f;
		d = Dot(n, vertices[0].p);
	}
	Vec3 centre() const
	{
		return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.0f;
	}
	// Add code here
	bool rayIntersect(const Ray& r, float& t, float& u, float& v) const
	{
		Vec3 E1 = vertices[1].p - vertices[0].p;
		Vec3 E2 = vertices[2].p - vertices[0].p;
		Vec3 p = r.dir.cross(E2);
		float det = E1.dot(p);
		if (fabs(det) < EPSILON) { return false; }
		float invDet = 1 / det;
		Vec3 T = r.o - vertices[0].p;
		u = T.dot(p) * invDet;
		if (u < 0 || u > 1) { return false; }
		Vec3 q = T.cross(E1);
		v = r.dir.dot(q) * invDet;
		if (v < 0 || v > 1 || (u + v) > 1) { return false; }
		t = E2.dot(q) * invDet;
		return t >= 0;
	}

	void interpolateAttributes(const float alpha, const float beta, const float gamma, Vec3& interpolatedNormal, float& interpolatedU, float& interpolatedV) const
	{
		interpolatedNormal = vertices[0].normal * alpha + vertices[1].normal * beta + vertices[2].normal * gamma;
		interpolatedNormal = interpolatedNormal.normalize();
		interpolatedU = vertices[0].u * alpha + vertices[1].u * beta + vertices[2].u * gamma;
		interpolatedV = vertices[0].v * alpha + vertices[1].v * beta + vertices[2].v * gamma;
	}
	// Add code here
	Vec3 sample(Sampler* sampler, float& pdf)
	{
		float r1 = sampler->next();
		float r2 = sampler->next();

		float sqrtR1 = sqrt(r1);
		float b0 = 1.0f - sqrtR1;
		float b1 = r2 * sqrtR1;
		float b2 = 1.0f - b0 - b1;

		pdf = 1.0f / area;

		return vertices[0].p * b0 + vertices[1].p * b1 + vertices[2].p * b2;
	}
	Vec3 gNormal()
	{
		return (n * (Dot(vertices[0].normal, n) > 0 ? 1.0f : -1.0f));
	}
};

class AABB
{
public:
	Vec3 max;
	Vec3 min;
	AABB()
	{
		reset();
	}
	void reset()
	{
		max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	void extend(const Vec3 p)
	{
		max = Max(max, p);
		min = Min(min, p);
	}
	void extend(const AABB b)
	{
		max = Max(max, b.max);
		min = Min(min, b.min);
	}
	void extend(const Triangle t)
	{
		this->extend(t.vertices[0].p);
		this->extend(t.vertices[1].p);
		this->extend(t.vertices[2].p);
	}
	// Add code here
	bool orayAABB(const Ray& r, float& t)
	{
		float tentry = cmax(cmax((min.x - r.o.x) * r.invDir.x, (min.y - r.o.y) * r.invDir.y), (min.z - r.o.z) * r.invDir.z);
		float texit = cmin(cmin((max.x - r.o.x) * r.invDir.x, (max.y - r.o.y) * r.invDir.y), (max.z - r.o.z) * r.invDir.z);
		if (tentry < texit && texit >= 0 && tentry < 0 && t >= tentry)
			return true;
		return false;
	}
	bool rayAABB(const Ray& r, float& t)
	{
		float tmin = -FLT_MAX;
		float tmax = FLT_MAX;

		if (std::fabs(r.dir.x) < 1e-8f) {
			if (r.o.x < min.x || r.o.x > max.x)
				return false;
		}
		else {
			float invD = 1.0f / r.dir.x;
			float t0 = (min.x - r.o.x) * invD;
			float t1 = (max.x - r.o.x) * invD;
			if (invD < 0.0f)
				std::swap(t0, t1);
			tmin = std::max(tmin, t0);
			tmax = std::min(tmax, t1);
			if (tmax < tmin)
				return false;
		}

		if (std::fabs(r.dir.y) < 1e-8f) {
			if (r.o.y < min.y || r.o.y > max.y)
				return false;
		}
		else {
			float invD = 1.0f / r.dir.y;
			float t0 = (min.y - r.o.y) * invD;
			float t1 = (max.y - r.o.y) * invD;
			if (invD < 0.0f)
				std::swap(t0, t1);
			tmin = std::max(tmin, t0);
			tmax = std::min(tmax, t1);
			if (tmax < tmin)
				return false;
		}

		if (std::fabs(r.dir.z) < 1e-8f) {
			if (r.o.z < min.z || r.o.z > max.z)
				return false;
		}
		else {
			float invD = 1.0f / r.dir.z;
			float t0 = (min.z - r.o.z) * invD;
			float t1 = (max.z - r.o.z) * invD;
			if (invD < 0.0f)
				std::swap(t0, t1);
			tmin = std::max(tmin, t0);
			tmax = std::min(tmax, t1);
			if (tmax < tmin)
				return false;
		}

		t = tmin;
		if (t < 0.0f) {
			t = tmax;
			if (t < 0.0f)
				return false;
		}
		return true;
	}

	// Add code here
	bool rayAABB(const Ray& r)
	{
		float t;
		return rayAABB(r, t);
	}
	// Add code here
	float area()
	{
		Vec3 size = max - min;
		return ((size.x * size.y) + (size.y * size.z) + (size.x * size.z)) * 2.0f;
	}
	bool rayAABB(const Ray& r, float& tEnter, float& tExit) const
	{

		tEnter = -FLT_MAX;
		tExit = FLT_MAX;

		float invDirX = 1.0f / r.dir.x;
		float tx1 = (min.x - r.o.x) * invDirX;
		float tx2 = (max.x - r.o.x) * invDirX;
		if (tx1 > tx2) std::swap(tx1, tx2);
		tEnter = (tx1 > tEnter) ? tx1 : tEnter;
		tExit = (tx2 < tExit) ? tx2 : tExit;
		if (tExit < tEnter) return false;

		float invDirY = 1.0f / r.dir.y;
		float ty1 = (min.y - r.o.y) * invDirY;
		float ty2 = (max.y - r.o.y) * invDirY;
		if (ty1 > ty2) std::swap(ty1, ty2);
		tEnter = (ty1 > tEnter) ? ty1 : tEnter;
		tExit = (ty2 < tExit) ? ty2 : tExit;
		if (tExit < tEnter) return false;

		float invDirZ = 1.0f / r.dir.z;
		float tz1 = (min.z - r.o.z) * invDirZ;
		float tz2 = (max.z - r.o.z) * invDirZ;
		if (tz1 > tz2) std::swap(tz1, tz2);
		tEnter = (tz1 > tEnter) ? tz1 : tEnter;
		tExit = (tz2 < tExit) ? tz2 : tExit;
		if (tExit < tEnter) return false;

		return true;
	}

};

class Sphere
{
public:
	Vec3 centre;
	float radius;
	void init(Vec3& _centre, float _radius)
	{
		centre = _centre;
		radius = _radius;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		return false;
	}
};

struct IntersectionData
{
	unsigned int ID;
	float t;
	float alpha;
	float beta;
	float gamma;
};

#define MAXNODE_TRIANGLES 8
#define TRAVERSE_COST 1.0f
#define TRIANGLE_COST 2.0f
#define BUILD_BINS 32

class BVHNode
{
public:
	AABB bounds;
	BVHNode* r;
	BVHNode* l;
	

	int startIndex;
	int endIndex;

	BVHNode()
		: l(nullptr)
		, r(nullptr)
		, startIndex(0)
		, endIndex(0)
	{
	}

	static AABB triangleBounds(const Triangle& tri)
	{
		AABB box;
		box.reset();
		box.extend(tri.vertices[0].p);
		box.extend(tri.vertices[1].p);
		box.extend(tri.vertices[2].p);
		return box;
	}

	void buildRecursive(std::vector<Triangle>& triangles, int start, int end)
	{
		bounds.reset();
		for (int i = start; i < end; i++) {
			bounds.extend(triangles[i].vertices[0].p);
			bounds.extend(triangles[i].vertices[1].p);
			bounds.extend(triangles[i].vertices[2].p);
		}

		int numTriangles = end - start;
		if (numTriangles <= MAXNODE_TRIANGLES)
		{
			this->startIndex = start;
			this->endIndex = end;
			return;
		}


		Vec3 size = bounds.max - bounds.min;
		int axis = 0;
		if (size.y > size.x && size.y > size.z)
			axis = 1;
		else if (size.z > size.x && size.z > size.y)
			axis = 2;

		std::sort(triangles.begin() + start, triangles.begin() + end,
			[axis](const Triangle& a, const Triangle& b) {
				float ca = (axis == 0) ? a.centre().x : (axis == 1) ? a.centre().y : a.centre().z;
				float cb = (axis == 0) ? b.centre().x : (axis == 1) ? b.centre().y : b.centre().z;
				return ca < cb;
			});

		int n = numTriangles;
		std::vector<AABB> leftBounds(n);
		std::vector<AABB> rightBounds(n);

		leftBounds[0] = triangleBounds(triangles[start]);
		for (int i = 1; i < n; i++) {
			leftBounds[i] = leftBounds[i - 1];
			leftBounds[i].extend(triangleBounds(triangles[start + i]));
		}
		rightBounds[n - 1] = triangleBounds(triangles[end - 1]);
		for (int i = n - 2; i >= 0; i--) {
			rightBounds[i] = rightBounds[i + 1];
			rightBounds[i].extend(triangleBounds(triangles[start + i]));
		}

		float totalArea = bounds.area();
		float bestCost = FLT_MAX;
		int bestSplit = -1;

		for (int i = 1; i < n; i++) {
			float leftArea = leftBounds[i - 1].area();
			float rightArea = rightBounds[i].area();
			int leftCount = i;
			int rightCount = n - i;
			float cost = 1.0f + (leftArea * leftCount + rightArea * rightCount) / totalArea;
			if (cost < bestCost) {
				bestCost = cost;
				bestSplit = i;
			}
		}

		if (bestCost >= static_cast<float>(numTriangles)) {
			this->startIndex = start;
			this->endIndex = end;
			return;
		}

		int mid = start + bestSplit;
		l = new BVHNode();
		r = new BVHNode();
		l->buildRecursive(triangles, start, mid);
		r->buildRecursive(triangles, mid, end);
	}

	void build(std::vector<Triangle>& inputTriangles, std::vector<Triangle>& triangles)
	{
		triangles = inputTriangles;
		buildRecursive(triangles, 0, static_cast<int>(triangles.size()));
	}

	void traverse(const Ray& ray, const std::vector<Triangle>& triangles, IntersectionData& intersection)
	{
		float tBox;
		if (!bounds.rayAABB(ray, tBox)) {
			return;
		}
		if (!l && !r)
		{
			for (int i = startIndex; i < endIndex; i++)
			{
				float t, u, v;
				if (triangles[i].rayIntersect(ray, t, u, v) && t > 1e-4f && t < intersection.t)
				{
					intersection.t = t;
					intersection.alpha = 1 - u - v;
					intersection.beta = u;
					intersection.gamma = v;
					intersection.ID = i;
				}
			}
			return;
		}
		if (l) l->traverse(ray, triangles, intersection);
		if (r) r->traverse(ray, triangles, intersection);
	}

	IntersectionData traverse(const Ray& ray, const std::vector<Triangle>& triangles)
	{
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		traverse(ray, triangles, intersection);
		return intersection;
	}

	bool traverseVisible(const Ray& ray, const std::vector<Triangle>& triangles, float maxT)
	{
		float tBox;
		float tMin, tMax;
		if (!bounds.rayAABB(ray, tMin, tMax)) {
			return true;
		}
		if (tMin > maxT) {
			return true;
		}
		if (!l && !r)
		{
			for (int i = startIndex; i < endIndex; i++)
			{
				float t, u, v;
				if (triangles[i].rayIntersect(ray, t, u, v) && t > 1e-4f && t < maxT)
				{
					return false;
				}
			}
			return true;
		}
		bool leftVis = l ? l->traverseVisible(ray, triangles, maxT) : true;
		bool rightVis = r ? r->traverseVisible(ray, triangles, maxT) : true;
		return leftVis && rightVis;
	}

};
