# Project 3: WebGL Procedural Shaders

## Versions

### [1. Wireframe Triangle](triangle.html)
A simple equilateral triangle rendered as a wireframe. Uses `gl_VertexID` and trigonometry to procedurally generate three vertices evenly spaced around a circle. Rendered using `gl.LINE_LOOP`.

### [2. Filled Polygon](polygon.html)
A 10-sided convex polygon rendered as a filled shape. Introduces uniform variable `N` to control the number of vertices dynamically. Uses `gl.TRIANGLE_FAN` to create a solid disk-like shape.

### [3. Five-Pointed Star](star.html)
A filled yellow five-pointed star. This version modifies vertex positions based on even or odd `gl_VertexID` to create altering outer tips with a radius of 1.0 and inner indentations of radius 0.5.

### [4. Colorful Spinning Star](spinningstar.html)
The same five-pointed star, now animated to rotate continuously and with a blue purple pink color. This version introduces the variable `t` for time, which is added to the angle calculation each frame to create smooth rotation.