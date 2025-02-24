import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image


def load_texture(filename, texture_cache):
    if filename in texture_cache:
        return texture_cache[filename]

    img = Image.open(filename).convert("RGB")
    img_data = np.array(img, dtype=np.uint8)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        img.width,
        img.height,
        0,
        GL_RGB,
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
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 1),
        (0, 0),
        (0, 1),
        (0, 0),
        (1, 0),
        (1, 1),
        (1, 1),
        (0, 1),
        (0, 0),
        (1, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (0, 0),
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
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


def render_sphere(x, y, z, radius, slices=16, stacks=16, texture=None):
    glPushMatrix()
    glTranslatef(x, y, z)

    if texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)

    quadric = gluNewQuadric()
    gluQuadricTexture(quadric, GL_TRUE)
    gluSphere(quadric, radius, slices, stacks)
    gluDeleteQuadric(quadric)

    if texture is not None:
        glDisable(GL_TEXTURE_2D)

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
