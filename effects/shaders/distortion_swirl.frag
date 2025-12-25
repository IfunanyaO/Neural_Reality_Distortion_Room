#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D u_tex;
uniform float u_time;
uniform float u_intensity;  // 0–1
uniform float u_audio;      // 0–1

void main() {
    // Centered UV
    vec2 uv = vTexCoord - 0.5;

    float r = length(uv);
    float angle = atan(uv.y, uv.x);

    // Swirl amount reacts to both slider + audio
    float swirl_base = mix(0.0, 3.5, u_intensity);
    float swirl_audio = u_audio * 2.5;
    float swirl = swirl_base + swirl_audio;

    // Radial falloff so center swirls more
    float falloff = exp(-r * 4.0);

    // Animate subtly over time
    angle += swirl * falloff + 0.2 * sin(u_time * 0.5) * falloff;

    vec2 warped = vec2(cos(angle), sin(angle)) * r;
    vec2 coord = warped + 0.5;

    vec3 color = texture(u_tex, coord).rgb;

    FragColor = vec4(color, 1.0);
}
