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

    // Base pulse speed
    float speed = 2.0;
    float pulse = sin(u_time * speed) * 0.5 + 0.5;

    // Use audio as a boost
    float energy = pulse + u_audio;

    // Zoom pulse
    float zoom_amount = mix(0.0, 0.2, u_intensity) * energy;

    // Radial zoom (breathing)
    vec2 uv_zoomed = center + delta * (1.0 - zoom_amount);

    // Soft halo glow
    float halo = smoothstep(0.5, 0.0, abs(dist - 0.3));
    halo *= energy * 0.4;

    vec3 base = texture(u_tex, uv_zoomed).rgb;

    // Add subtle tint based on pulse
    vec3 tint = vec3(0.6, 0.8, 1.0); // cool neural-ish
    vec3 color = mix(base, base * tint, halo);

    FragColor = vec4(color, 1.0);
}
