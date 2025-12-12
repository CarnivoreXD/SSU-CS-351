# Project 3: WebGL Procedural Shaders

A series of WebGL applications that progressively build upon each other, demonstrating procedural geometry generation using vertex and fragment shaders.

## Versions

### [1. Wireframe Triangle](triangle.html)
A simple equilateral triangle rendered as a wireframe. Uses `gl_VertexID` and trigonometry to procedurally generate three vertices evenly spaced around a circle. Rendered using `gl.LINE_LOOP`.

### [2. Filled Polygon](polygon.html)
A 10-sided convex polygon rendered as a filled shape. Introduces uniform variable `N` to control the number of vertices dynamically. Uses `gl.TRIANGLE_FAN` to create a solid disk-like shape.

### [3. Five-Pointed Star](star.html)
A filled five-pointed star shape. Modifies vertex positions based on even/odd `gl_VertexID` to create alternating outer tips (radius 1.0) and inner indentations (radius 0.4). The center vertex is explicitly placed at the origin for the triangle fan.

### [4. Spinning Star](spinningstar.html)
The same five-pointed star, now animated to rotate continuously. Introduces uniform variable `t` for time, which is added to the angle calculation each frame to create smooth rotation.

### [5. Colorful Star](spinningstar.html) *(Extra Credit)*
The rotating star with a color gradient effect. Passes the `radius` value from the vertex shader to the fragment shader, creating a cyan-to-magenta gradient from the center to the tips.

## How to Run

Open any `.html` file directly in a WebGL-capable browser (Chrome, Firefox, Edge, Safari).

## Files

- `initShaders.js` - Helper function for compiling GLSL shaders
- `1-triangle.html` - Wireframe triangle
- `2-polygon.html` - Filled 10-sided polygon
- `3-star.html` - Five-pointed star
- `4-spinning.html` - Rotating star
- `5-colorful.html` - Colorful rotating star (extra credit)