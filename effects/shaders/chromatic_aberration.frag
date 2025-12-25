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

    // Base separation
    float base_sep = mix(0.0, 0.01, u_intensity);
    // Audio adds extra glitch burst
    float audio_sep = u_audio * 0.015;
    float sep = base_sep + audio_sep;

    // Animated flicker
    sep *= (0.7 + 0.3 * sin(u_time * 2.0));

    // Directions
    vec2 dir = (dist > 0.0001) ? normalize(delta) : vec2(0.0);

    // Offset each channel differently
    vec2 uv_r = uv + dir * sep;
    vec2 uv_g = uv;
    vec2 uv_b = uv - dir * sep;

    float r = texture(u_tex, uv_r).r;
    float g = texture(u_tex, uv_g).g;
    float b = texture(u_tex, uv_b).b;

    vec3 color = vec3(r, g, b);

    // Slight vignette to make center feel deeper
    float vignette = smoothstep(0.9, 0.4, dist);
    color *= vignette;

    FragColor = vec4(color, 1.0);
}
