#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D u_tex;
uniform float u_time;
uniform float u_intensity;  // 0–1
uniform float u_audio;      // 0–1

void main() {
    vec2 uv = vTexCoord;
    vec2 center = vec2(0.5);

    vec2 delta = uv - center;
    float dist = length(delta);

    // Ripple frequency & amplitude
    float freq = mix(4.0, 14.0, u_intensity);
    float amp  = mix(0.0, 0.03, u_intensity);

    // Audio pumps extra amplitude
    amp += u_audio * 0.04;

    // Time animated wave
    float wave = sin(dist * freq - u_time * 3.0);

    // Radial falloff so outer edges stay calmer
    float falloff = exp(-dist * 3.0);

    float offset = wave * amp * falloff;

    // Push along radial direction
    vec2 dir = (dist > 0.0001) ? (delta / dist) : vec2(0.0);
    vec2 coord = uv + dir * offset;

    vec3 color = texture(u_tex, coord).rgb;
    FragColor = vec4(color, 1.0);
}
