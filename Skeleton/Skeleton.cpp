//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;		//a kamera ablak egy pixel pontja, amire épp nézünk

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

//------------------------------------------------

//------------------------------------------------

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	const vec3 La = vec3(0.5f, 0.6f, 0.6f);
	const vec3 Le = vec3(0.8f, 0.8f, 0.8f);
	const vec3 lightPosition = vec3(0.4f, 0.4f, 0.25f);
	const vec3 ka = vec3(0.5f, 0.5f, 0.5f);
	const float shininess = 500.0f;
	const int maxdepth = 5;
	const float epsilon = 0.01f;

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	// smooth?
	};

	struct Ray {
		vec3 start, dir, weight;
	};

	const int objFaces = 12;

	uniform vec3 wEye;
	uniform vec3 v[20];
	uniform int planes[objFaces * 3];
	uniform vec3 kd[2], ks[2], F0Gold, F0PerfectReflect;

	void getObjPlane(int i, float scale,  out vec3 p,  out vec3 normal) {
		vec3 p1 = v[planes[3 * i] - 1];
		vec3 p2 = v[planes[3 * i + 1] - 1];
		vec3 p3 = v[planes[3 * i + 2] - 1];
		normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0) normal = -normal;
		p = p1 * scale + vec3(0, 0, 0.03f);
	}

	Hit intersectConvexPolyhedron(Ray ray, Hit hit, float scale, int mat) {
		for (int i = 0; i < objFaces; i++)
		{
			vec3 p1;
			vec3 normal;
			getObjPlane(i, scale, p1, normal);
			float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;
			vec3 intersect = ray.start + ray.dir * ti;
			bool outside = false;
			bool closeToEdge = false;
			for (int j = 0; j < objFaces; j++) {
				if (i == j) continue;
				vec3 p11, n;
				getObjPlane(j, scale, p11, n);
				if (dot(n, intersect - p11) > 0) {
					outside = true;
					break;
				}
				if (dot(n, intersect - p11) > -0.15f && dot(n, intersect - p11) <= 0.0f) {
					hit.mat = 0;			//ha közel vagyunk a dedokaéder éleihez akkor változtassuk a materialt 0-ásra (diffúz szürke)
					closeToEdge = true;
				}
			}
			if (!outside) {
				hit.t = ti;
				hit.position = intersect;
				hit.normal = normalize(normal);
				if(!closeToEdge)
					hit.mat = mat;
			}
		}
		return hit;
	}

	Hit intersectSphere(float radius, vec3 center, Ray ray, int material) {		//a kapott sugár hol metszi el ezt a gömböt
		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4 * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2 / a;
		float t2 = (-b - sqrt_discr) / 2 / a;
		if (t1 <= 0) return hit; // t1 >= t2 for sure
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) / radius;		//gömb esetén így számíthatjuk ki a normálvektort adott pontban (de csak gömbnél)
		hit.mat = material;
		return hit;
	}


	vec3 gradiens(float a, float b, float c, vec3 position) {
		float dx = 2 * a * position.x * exp(a * pow(position.x, 2) - c * position.z + b * pow(position.y, 2));
		float dy = 2 * b * position.y * exp(b * pow(position.y, 2) - c * position.z + a * pow(position.x, 2));
		float dz = -c * exp(-c * position.z + b * pow(position.y, 2) + a * pow(position.x, 2));
		return vec3(dx, dy, dz);
	}

	Hit intersectImplicitSurface(float a, float b, float c, Ray ray, int material) {
		//a felület implicit egyenletébe behelyettesítjük a sugár egyenletét
		//a* (ray.start.x + ray.dir.x * t_metszes) ^ 2 + b * (ray.start.y + ray.dir.y * t_metszes) ^ 2 - c * (ray.start.z + ray.dir.z * t_metszes) = 0
		//fenti másodfokú egyenletben t_metszes az ismeretlen (azon t idöpillanat amikor a sugár elmetszi ezen felületet)
		//megoldani t_metszes-re, majd a kisebbik (de nagyobb nulla) t_metszes-t visszahelyettesíteni: hit.position = ray.start + ray.dir * t_metszes egyenletbe
		//ebböl kapunk egymetszéspontot,amire meg kell nézni h belefér-e a megadott körbe -> 0,0,0 középpontú körtöl 0.3-nál kisebb távolságra van-e
		//ha nem akkor eldobjuk, azaz hit.t = -1-et állítjuk be

		Hit hit;
		hit.t = -1;
		float t_metszes;
		float apar, bpar, cpar;		//a másodfokú egyenlet megoldóképletének együtthatói
		apar = a * pow(ray.dir.x, 2) + b * pow(ray.dir.y, 2);
		bpar = a * 2 * ray.start.x * ray.dir.x + b * 2 * ray.start.y * ray.dir.y + c * ray.dir.z;
		cpar = a * pow(ray.start.x, 2) + b * pow(ray.start.y, 2) + c * ray.start.z;
		float discr = bpar * bpar - 4 * apar * cpar;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-bpar + sqrt_discr) / 2 / apar;
		float t2 = (-bpar - sqrt_discr) / 2 / apar;
		if (t1 <= 0) return hit;
		t_metszes = (t2 > 0) ? t2 : t1;
		vec3 position = ray.start + ray.dir * t_metszes;
		vec3 distVector = position - vec3(0.0f, 0.0f, 0.0f);
		float dist = length(distVector);
		if (dist > 0.3) return hit;
		hit.t = t_metszes;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(gradiens(a, b, c, position));
		hit.mat = material;
		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		Hit tempHit;
		bestHit.t = -1;
		tempHit.t = -1;
		
		//bestHit = intersectConvexPolyhedron(ray, bestHit, 0.06f, 2);
		//bestHit = intersectConvexPolyhedron(ray, bestHit, 1.0f, 2);
		bestHit = intersectConvexPolyhedron(ray, bestHit, 1.2f, 3);

		tempHit = intersectSphere(0.2f, vec3(-0.5f, 0.0f, 0.0f), ray, 2);
		if (tempHit.t > 0.0f && (bestHit.t < 0.0f || tempHit.t < bestHit.t))  bestHit = tempHit;

		tempHit = intersectImplicitSurface(10.5f, 10.5f, 1.5f, ray, 2);
		if (tempHit.t > 0.0f && (bestHit.t < 0.0f || tempHit.t < bestHit.t))  bestHit = tempHit;

		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) {
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}



	vec3 trace(Ray ray) {
		//vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		for (int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) break;
			if (hit.mat < 2) {		//rough surface
				vec3 lightdir = normalize(lightPosition - hit.position);
				float cosTheta = dot(hit.normal, lightdir);
				if (cosTheta > 0) {
					vec3 LeIn = Le / dot(lightPosition - hit.position, lightPosition - hit.position);
					outRadiance = outRadiance + (ray.weight * LeIn * kd[hit.mat] * cosTheta);
					vec3 halfway = normalize(-ray.dir + lightdir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + (ray.weight * LeIn * ks[hit.mat] * pow(cosDelta, shininess));
				}
				ray.weight = ray.weight * ka;
				break;
			}
			//mirror reflection
			if (hit.mat == 2) {
				ray.weight = ray.weight * (F0Gold + (vec3(1, 1, 1) - F0Gold) * pow(dot(-ray.dir, hit.normal), 5));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			}
			if (hit.mat == 3) {
				ray.weight = ray.weight * (F0PerfectReflect + (vec3(1, 1, 1) - F0PerfectReflect) * pow(dot(-ray.dir, hit.normal), 5));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			}
		}
		outRadiance = outRadiance + (ray.weight * La);
		return outRadiance;
	}

	in  vec3 p;
	out vec4 fragmentColor;

	void main() {
		Ray ray;
		ray.start = wEye;
		ray.dir = normalize(p - wEye);
		ray.weight = vec3(1, 1, 1);
		fragmentColor = vec4(trace(ray), 1);
	}
)";

//---------------------------
struct Camera {
	//---------------------------
	vec3 eye, lookat, right;
	vec3 pvup;	//preferált függöleges irány
	vec3 rvup;	//valódi függöleges irány
	float fov = 45 * (float)M_PI / 180;

	Camera() : eye(1, 0, 0), pvup(0, 0, -1), lookat(0, 0, 0) {
		set();
	}
	void set() {
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(pvup, w)) * f * tanf(fov / 2);
		rvup = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float t) {		// a szempozíciót egy körpályán mozgatjuk
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
		set();
	}

	void Step(float step) {		// fölfelé v lefelé lépünk egy kicsit
		eye = normalize(eye + pvup * step) * length(eye);
		set();
	}
};

//---------------------------
struct Light {
	//---------------------------
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
};

GPUProgram shader;
Camera camera;
bool animate = true;

float Fresnel(float n, float kappa) {
	return ((n - 1) * (n - 1) + kappa * kappa) / ((n + 1) * (n + 1) + kappa * kappa);
}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	unsigned int vao;
	glGenVertexArrays(1, &vao);	// create 1 vertex array object
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;
	glGenBuffers(1, &vbo);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

	// create program for the GPU
	shader.create(vertexSource, fragmentSource, "fragmentColor");

	const float g = 0.618f;
	const float G = 1.618f;
	std::vector<vec3> v = {
		vec3(0,g,G), vec3(0,-g,G), vec3(0,-g,-G), vec3(0,g,-G),
		vec3(G,0,g), vec3(-G,0,g), vec3(-G,0,-g), vec3(G,0,-g),
		vec3(g,G,0), vec3(-g,G,0), vec3(-g,-G,0), vec3(g,-G,0),
		vec3(1,1,1), vec3(-1,1,1), vec3(-1,-1,1), vec3(1,-1,1),
		vec3(1,-1,-1), vec3(1,1,-1), vec3(-1,1,-1), vec3(-1,-1,-1)
	};
	for (int i = 0; i < v.size(); i++) {
		shader.setUniform(v[i], "v[" + std::to_string(i) + "]");
	}

	std::vector<int> planes = {
		1,2,16,
		1,13,9,
		1,14,6,
		2,15,11,
		3,4,18,
		3,17,12,
		3,20,7,
		19,10,9,
		16,12,17,
		5,8,18,
		14,10,19,
		6,7,20
	};
	for (int i = 0; i < planes.size(); i++) {
		shader.setUniform(planes[i], "planes[" + std::to_string(i) + "]");
	}

	//két diffúz anyagunk lesz (egy szürkés és egy sárgás) ezeknek itt adjuk meg a tulajdonságait
	shader.setUniform(vec3(0.1f, 0.2f, 0.3f), "kd[0]");
	shader.setUniform(vec3(1.5f, 0.6f, 0.4f), "kd[1]");
	shader.setUniform(vec3(5, 5, 5), "ks[0]");
	shader.setUniform(vec3(1, 1, 1), "ks[1]");

	//speifikáció szerinti arany törésmutatók és kioltási tényezök r,g,b hullámhosszokon
	float redFresnel = Fresnel(0.17, 3.1);
	float greenFresnel = Fresnel(0.35, 2.7);
	float blueFresnel = Fresnel(1.5, 1.9);
	shader.setUniform(vec3(redFresnel, greenFresnel, blueFresnel), "F0Gold");
	shader.setUniform(vec3(1, 1, 1), "F0PerfectReflect");		//tökéletes visszaverödést biztosító Fresnel mutató
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	shader.setUniform(camera.eye, "wEye");
	shader.setUniform(camera.lookat, "wLookAt");
	shader.setUniform(camera.right, "wRight");
	shader.setUniform(camera.rvup, "wUp");
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 't') shader.setUniform(1, "top");
	if (key == 'T') shader.setUniform(0, "top");
	if (key == 'f') camera.Step(0.1f);
	if (key == 'F') camera.Step(-0.1f);
	if (key == 'a') animate = !animate;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	if (animate) {
		camera.Animate(glutGet(GLUT_ELAPSED_TIME) / 1000.0f);
	}
	glutPostRedisplay();
}