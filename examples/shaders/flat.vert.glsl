#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 colour;
layout(std140) uniform Frame { vec4 tint; };
out vec4 v_colour;
void main() { v_colour = colour * tint; gl_Position = vec4(position, 0.0, 1.0); }
