import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image


def load_texture(filename, texture_cache):
    if filename in texture_cache:
        return texture_cache[filename]

    img = Image.open(filename).convert("RGBA")
    img_data = np.array(img, dtype=np.uint8)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        img.width,
        img.height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        img_data,
    )
    glGenerateMipmap(GL_TEXTURE_2D)

    texture_cache[filename] = texture_id
    return texture_id


def render_plane(x, y, z, area, texture_id, repeat=None):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    half_area = area / 2.0
    # If repeat is not specified, make each texture unit correspond to one grid unit
    if repeat is None:
        repeat = area

    glPushMatrix()
    glTranslatef(x, y, z)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex3f(-half_area, 0.0, half_area)
    glTexCoord2f(repeat, 0)
    glVertex3f(half_area, 0.0, half_area)
    glTexCoord2f(repeat, repeat)
    glVertex3f(half_area, 0.0, -half_area)
    glTexCoord2f(0, repeat)
    glVertex3f(-half_area, 0.0, -half_area)
    glEnd()
    glPopMatrix()

    glDisable(GL_TEXTURE_2D)


def render_cube(x, y, z, texture=None, color=None):
    glPushMatrix()
    glTranslatef(x, y, z)

    if texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
    elif color is not None:
        glDisable(GL_TEXTURE_2D)
        glColor3f(*color)

    vertices = [
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5),
        (0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, 0.5, -0.5),
    ]

    tex_coords = [
        (0, 1),
        (1, 1),
        (1, 0),
        (0, 0),  # Flipped v-coordinate
        (1, 1),
        (1, 0),
        (0, 0),
        (0, 1),  # Flipped v-coordinate
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),  # Flipped v-coordinate
        (1, 0),
        (0, 0),
        (0, 1),
        (1, 1),  # Flipped v-coordinate
        (1, 1),
        (1, 0),
        (0, 0),
        (0, 1),  # Flipped v-coordinate
        (0, 1),
        (1, 1),
        (1, 0),
        (0, 0),  # Flipped v-coordinate
    ]

    glBegin(GL_QUADS)
    for i in range(0, len(vertices), 4):
        for j in range(4):
            if texture is not None:
                glTexCoord2f(*tex_coords[i + j])
            glVertex3f(*vertices[i + j])
    glEnd()

    if texture is not None:
        glDisable(GL_TEXTURE_2D)
    elif color is not None:
        glColor3f(1.0, 1.0, 1.0)  # Reset color to white
    glPopMatrix()


def render_sphere(x, y, z, radius, slices=16, stacks=16, texture=None, color=None):
    glPushMatrix()
    glTranslatef(x, y, z)

    if texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
        quadric = gluNewQuadric()
        gluQuadricTexture(quadric, GL_TRUE)
    elif color is not None:
        glDisable(GL_TEXTURE_2D)
        glColor3f(*color)
        quadric = gluNewQuadric()
    else:
        # Default case if neither texture nor color is specified
        # (You might want to set a default color or ensure texture is disabled)
        glDisable(GL_TEXTURE_2D)  # Ensure texturing is off if no texture provided
        glColor3f(1.0, 1.0, 1.0)  # Default to white if nothing specified
        quadric = gluNewQuadric()

    gluSphere(quadric, radius, slices, stacks)
    gluDeleteQuadric(quadric)

    if texture is not None:
        glDisable(GL_TEXTURE_2D)
    elif color is not None:
        glColor3f(1.0, 1.0, 1.0)  # Reset color to white

    glPopMatrix()


def render_cylinder(
    x, y, z, radius, height, texture=None, color=None, slices=32, stacks=1
):
    # Render a cylinder at position (x,y,z) with given radius and height, aligned along the y-axis
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(-90, 1, 0, 0)  # rotate to align cylinder axis from z to y
    quadric = gluNewQuadric()

    if texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
        gluQuadricTexture(quadric, GL_TRUE)
    elif color is not None:
        glDisable(GL_TEXTURE_2D)
        glColor3f(*color)

    gluCylinder(quadric, radius, radius, height, slices, stacks)
    gluDeleteQuadric(quadric)

    if texture is not None:
        glDisable(GL_TEXTURE_2D)
    elif color is not None:
        glColor3f(1.0, 1.0, 1.0)  # Reset color to white

    glPopMatrix()


def render_torus(inner_radius, outer_radius, nsides=16, rings=30, texture=None):
    # Render a torus using GLUT's solid torus; inner_radius is the tube radius, outer_radius is the distance from center to tube center.
    from OpenGL.GLUT import glutSolidTorus

    glPushMatrix()
    if texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
        # Enable texture generation using sphere mapping
        glEnable(GL_TEXTURE_GEN_S)
        glEnable(GL_TEXTURE_GEN_T)
        glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
        glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)

    glutSolidTorus(inner_radius, outer_radius, nsides, rings)

    if texture is not None:
        # Disable texture generation
        glDisable(GL_TEXTURE_GEN_S)
        glDisable(GL_TEXTURE_GEN_T)
        glDisable(GL_TEXTURE_2D)
    glPopMatrix()


def render_diamond(x, y, z, size, texture=None, color=None):
    """Renders a diamond shape (two pyramids joined by a central cube)."""
    glPushMatrix()
    glTranslatef(x, y, z)
    glScalef(size, size, size)  # Apply overall size scaling

    if texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
    elif color is not None:
        glDisable(GL_TEXTURE_2D)
        glColor3f(*color)
    else:
        # Default color if neither is provided
        glDisable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 1.0)  # White

    # Define vertices
    pyramid_height = 0.5  # Height of each pyramid part
    cube_half_height = 0.2  # Half-height of the central cube
    base_half_width = 0.5

    apex_top = (0, pyramid_height + cube_half_height, 0)
    apex_bottom = (0, -(pyramid_height + cube_half_height), 0)

    # Vertices for the top base of the cube (and base of top pyramid)
    top_base = [
        (base_half_width, cube_half_height, base_half_width),  # Front-right
        (-base_half_width, cube_half_height, base_half_width),  # Front-left
        (-base_half_width, cube_half_height, -base_half_width),  # Back-left
        (base_half_width, cube_half_height, -base_half_width),  # Back-right
    ]

    # Vertices for the bottom base of the cube (and base of bottom pyramid)
    bottom_base = [
        (base_half_width, -cube_half_height, base_half_width),  # Front-right
        (-base_half_width, -cube_half_height, base_half_width),  # Front-left
        (-base_half_width, -cube_half_height, -base_half_width),  # Back-left
        (base_half_width, -cube_half_height, -base_half_width),  # Back-right
    ]

    # Define texture coordinates
    tex_coords_base = [
        (1.0, 1.0),  # fr
        (0.0, 1.0),  # fl
        (0.0, 0.0),  # bl
        (1.0, 0.0),  # br
    ]
    tex_coord_top = (0.5, 1.0)  # Texture coord for top apex
    tex_coord_bottom = (0.5, 0.0)  # Texture coord for bottom apex
    # Texture coords for cube sides (simple repeat)
    tex_coords_cube_side = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    # Render Top Pyramid
    glBegin(GL_TRIANGLES)
    for i in range(4):
        v1 = top_base[i]
        v2 = top_base[(i + 1) % 4]
        v3 = apex_top
        if texture is not None:
            glTexCoord2f(*tex_coords_base[i])
            glVertex3f(*v1)
            glTexCoord2f(*tex_coords_base[(i + 1) % 4])
            glVertex3f(*v2)
            glTexCoord2f(*tex_coord_top)
            glVertex3f(*v3)
        else:
            glVertex3f(*v1)
            glVertex3f(*v2)
            glVertex3f(*v3)
    glEnd()

    # Render Bottom Pyramid (reverse vertex order for correct face culling)
    glBegin(GL_TRIANGLES)
    for i in range(4):
        v1 = bottom_base[i]
        v2 = bottom_base[(i + 1) % 4]
        v3 = apex_bottom
        if texture is not None:
            glTexCoord2f(*tex_coords_base[i])
            glVertex3f(*v1)
            glTexCoord2f(*tex_coord_bottom)  # Use bottom apex tex coord
            glVertex3f(*v3)
            glTexCoord2f(*tex_coords_base[(i + 1) % 4])
            glVertex3f(*v2)
        else:
            glVertex3f(*v1)
            glVertex3f(*v3)
            glVertex3f(*v2)
    glEnd()

    # Render Cube Sides
    glBegin(GL_QUADS)
    for i in range(4):
        v_bl = bottom_base[i]  # Bottom-left of quad
        v_br = bottom_base[(i + 1) % 4]  # Bottom-right of quad
        v_tr = top_base[(i + 1) % 4]  # Top-right of quad
        v_tl = top_base[i]  # Top-left of quad

        if texture is not None:
            glTexCoord2f(*tex_coords_cube_side[0])
            glVertex3f(*v_bl)
            glTexCoord2f(*tex_coords_cube_side[1])
            glVertex3f(*v_br)
            glTexCoord2f(*tex_coords_cube_side[2])
            glVertex3f(*v_tr)
            glTexCoord2f(*tex_coords_cube_side[3])
            glVertex3f(*v_tl)
        else:
            glVertex3f(*v_bl)
            glVertex3f(*v_br)
            glVertex3f(*v_tr)
            glVertex3f(*v_tl)
    glEnd()

    if texture is not None:
        glDisable(GL_TEXTURE_2D)
    elif color is not None:
        glColor3f(1.0, 1.0, 1.0)  # Reset color

    glPopMatrix()


def render_shadow(x, z, radius, texture_id, intensity=0.5):
    """Renders a simple, circular shadow quad on the ground plane."""
    glPushMatrix()
    # Slightly raise the shadow to prevent z-fighting with the floor
    glTranslatef(x, -0.5 + 0.01, z)
    glScalef(radius, 1.0, radius)  # Scale the unit quad to the desired radius

    # Enable blending for transparency
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Use the shadow texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Set shadow color and intensity (alpha)
    # We use white color because the texture itself is dark.
    # Alpha controls how dark the shadow appears.
    glColor4f(1.0, 1.0, 1.0, intensity)

    # Render a quad facing up
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex3f(-1.0, 0.0, 1.0)  # Bottom-left
    glTexCoord2f(1, 0)
    glVertex3f(1.0, 0.0, 1.0)  # Bottom-right
    glTexCoord2f(1, 1)
    glVertex3f(1.0, 0.0, -1.0)  # Top-right
    glTexCoord2f(0, 1)
    glVertex3f(-1.0, 0.0, -1.0)  # Top-left
    glEnd()

    # Clean up GL state
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color to opaque white

    glPopMatrix()
