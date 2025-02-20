import glfw
import numpy as np
import os
import copy
from OpenGL.GL import *
from OpenGL.GLU import *
from sgw.utils.gl_utils import (
    load_texture,
    render_plane,
    render_cube,
    render_sphere,
    render_cylinder,
    render_torus,
)
from sgw.renderers.rend_interface import RendererInterface
from sgw.utils.base_utils import resize_obs
from gym import spaces


class Grid3DRenderer(RendererInterface):
    def __init__(self, resolution=128, torch_obs=False, field_of_view=None):
        self.resolution = resolution
        self.last_objects = None
        self.texture_cache = {}  # Add texture cache as an instance variable
        self.torch_obs = torch_obs
        self.initialize_glfw()
        if field_of_view is not None:
            self.configure_opengl(fov=field_of_view * 60)
        else:
            self.configure_opengl(fov=90.0)

        # Make the OpenGL context current before loading textures
        glfw.make_context_current(self.window)
        self.load_textures()
        self.display_lists = {}

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space for visual observations."""
        if self.torch_obs:
            # PyTorch format (channels, height, width)
            return spaces.Box(0, 1, shape=(3, 64, 64))
        # Standard format (height, width, channels)
        return spaces.Box(0, 1, shape=(self.resolution, self.resolution, 3))

    def initialize_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(
            self.resolution, self.resolution, "Offscreen", None, None
        )
        if not self.window:
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        self.width, self.height = self.resolution, self.resolution

    def configure_opengl(self, fov=120.0):
        glViewport(0, 0, self.width, self.height)
        glClearColor(135 / 255, 206 / 255, 235 / 255, 1.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def load_textures(self):
        glfw.make_context_current(self.window)  # Ensure context is current

        self.tex_folder = os.path.join(os.path.dirname(__file__), "textures/")
        self.textures = {
            name: load_texture(f"{self.tex_folder}{file}", self.texture_cache)
            for name, file in {
                "floor": "floor.png",
                "wall": "wall.png",
                "gem": "gem.png",
                "gem_bad": "gem_bad.png",
                "wood": "wood.png",
                "key": "key.png",
                "warp": "warp.png",
                "tree": "tree.png",
                "fruit": "fruit.png",
                "sign": "sign.png",
                "box": "wood.png",  # Reuse wood texture for box
            }.items()
        }

    def set_camera(self, agent_pos, agent_dir):
        offsets = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        offset = offsets[agent_dir]
        pos = (agent_pos[0], 0, agent_pos[1])
        target = (agent_pos[0] + offset[0], 0, agent_pos[1] + offset[1])
        up = (0, 1, 0)
        glLoadIdentity()
        gluLookAt(*pos, *target, *up)

    def create_scene_display_list(self, env):
        list_id = glGenLists(1)
        glNewList(list_id, GL_COMPILE)

        # Render walls
        if "walls" in env.objects:
            for wall in env.objects["walls"]:
                render_cube(wall.pos[0], 0.0, wall.pos[1], self.textures["wall"])

        # Render rewards
        if "rewards" in env.objects:
            for reward in env.objects["rewards"]:
                reward_val = (
                    reward.value[0] if isinstance(reward.value, list) else reward.value
                )
                texture = (
                    self.textures["gem"] if reward_val > 0 else self.textures["gem_bad"]
                )
                render_sphere(
                    reward.pos[0], -0.25, reward.pos[1], 0.25, texture=texture
                )

        # Render doors
        if "doors" in env.objects:
            for door in env.objects["doors"]:
                render_cube(door.pos[0], 0.0, door.pos[1], self.textures["wood"])

        # Render keys
        if "keys" in env.objects:
            for key in env.objects["keys"]:
                # Draw key handle using torus primitive
                glPushMatrix()
                glTranslatef(key.pos[0], -0.25, key.pos[1])
                glRotatef(90, 0, 0, 1)  # Rotate to orient torus in horizontal plane
                render_torus(
                    0.04, 0.15, nsides=16, rings=16, texture=self.textures["key"]
                )
                glPopMatrix()

                # Draw shaft using cylinder primitive
                glPushMatrix()
                glTranslatef(key.pos[0] + 0.15, -0.25, key.pos[1])
                glRotatef(90, 0, 0, 1)  # Rotate so that the cylinder lies horizontally
                render_cylinder(0, 0, 0, 0.05, 0.3, texture=self.textures["key"])
                glPopMatrix()

                # Draw teeth
                glPushMatrix()
                glTranslatef(key.pos[0] + 0.3, -0.25, key.pos[1])
                glScalef(0.1, 0.05, 0.15)
                render_cube(0, 0, 0, self.textures["key"])
                glPopMatrix()

        # Render warps
        if "warps" in env.objects:
            for warp in env.objects["warps"]:
                render_sphere(
                    warp.pos[0], -0.5, warp.pos[1], 0.33, texture=self.textures["warp"]
                )

        # Render trees
        if "trees" in env.objects:
            for tree in env.objects["trees"]:
                self._render_tree(tree.pos[0], tree.pos[1])

        # Render fruits
        if "fruits" in env.objects:
            for fruit in env.objects["fruits"]:
                render_sphere(
                    fruit.pos[0],
                    -0.3,
                    fruit.pos[1],
                    0.2,
                    texture=self.textures["fruit"],
                )

        # Render signs
        if "signs" in env.objects:
            for sign in env.objects["signs"]:
                self._render_sign(sign.pos[0], sign.pos[1])

        # Render boxes
        if "boxes" in env.objects:
            for box in env.objects["boxes"]:
                self._render_box(box.pos[0], box.pos[1])

        # Render other agents (excluding current agent)
        current_agent = (
            env.agents[self._current_agent_idx]
            if hasattr(self, "_current_agent_idx")
            else None
        )
        for i, agent in enumerate(env.agents):
            if agent is not None and agent != current_agent:
                render_cube(
                    agent.pos[0], -0.5, agent.pos[1], None, color=(0.5, 0.5, 0.5)
                )

        glEndList()
        return list_id

    def _render_tree(self, x, z):
        """Render a tree with a trunk and conical crown."""
        # Render trunk
        glPushMatrix()
        glTranslatef(x, -0.5, z)  # Start at ground level
        render_cylinder(0, 0, 0, 0.1, 1.0, texture=self.textures["tree"])
        glPopMatrix()

        # Render crown (cone-like shape made of cubes)
        crown_height = 1.5
        crown_layers = 3
        for i in range(crown_layers):
            layer_scale = 1.0 - (i / crown_layers)
            glPushMatrix()
            glTranslatef(x, 0.5 + (i * 0.5), z)  # Adjusted to start at top of trunk
            glScalef(layer_scale, 0.4, layer_scale)
            render_cube(0, 0, 0, self.textures["tree"])
            glPopMatrix()

    def _render_sign(self, x, z):
        """Render a sign with a post and board."""
        # Render post
        glPushMatrix()
        glTranslatef(x, -0.5, z)  # Start at ground level
        glScalef(0.1, 0.8, 0.1)
        render_cube(0, 0.5, 0, self.textures["wood"])
        glPopMatrix()

        # Render board
        glPushMatrix()
        glTranslatef(x, 0.3, z)  # Adjusted to be at appropriate height from ground
        glScalef(0.6, 0.4, 0.1)
        render_cube(0, 0, 0, self.textures["sign"])
        glPopMatrix()

    def _render_box(self, x, z):
        """Render a box as a cube with a lid line."""
        # Render main box
        glPushMatrix()
        glTranslatef(x, -0.25, z)  # Slightly raised from ground
        glScalef(0.8, 0.5, 0.8)  # Make it shorter than a full cube
        render_cube(0, 0, 0, self.textures["box"])
        glPopMatrix()

        # Render lid (thin rectangle on top)
        glPushMatrix()
        glTranslatef(x, 0.0, z)  # At the top of the box
        glScalef(0.8, 0.1, 0.8)  # Thin lid
        render_cube(0, 0, 0, self.textures["box"])
        glPopMatrix()

    def render_frame(self, env, agent_idx=0, is_state_view=False):
        glfw.make_context_current(self.window)  # Make context current
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Store current agent index for use in create_scene_display_list
        self._current_agent_idx = agent_idx

        # Set camera based on current agent
        agent = env.agents[agent_idx]
        self.set_camera(agent.pos, agent.looking)

        # Render all objects in the scene
        if "scene" not in self.display_lists or env.objects != self.last_objects:
            if "scene" in self.display_lists:
                glDeleteLists(self.display_lists["scene"], 1)
            self.display_lists["scene"] = self.create_scene_display_list(env)
            self.last_objects = copy.deepcopy(env.objects)
        glCallList(self.display_lists["scene"])

        # Render floor
        render_plane(
            env.grid_width / 2 - 0.5,  # Shift by half a unit to align with grid
            -0.5,
            env.grid_height / 2 - 0.5,  # Shift by half a unit to align with grid
            max(env.grid_width, env.grid_height),
            self.textures["floor"],
            repeat=max(env.grid_width, env.grid_height)
            / 4,  # Make each square in the 4x4 pattern correspond to one grid unit
        )

        # Read pixels
        buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        image = np.flip(image, axis=0)  # flip vertically
        image = np.flip(image, axis=1)  # flip horizontally
        return resize_obs(image, self.resolution, self.torch_obs)

    def render(self, env, agent_idx=0, is_state_view=False):
        # Implement the common interface render method
        return self.render_frame(env, agent_idx, is_state_view)

    def close(self):
        glfw.make_context_current(self.window)  # Make context current
        for list_id in self.display_lists.values():
            glDeleteLists(list_id, 1)
        glfw.destroy_window(self.window)

        # Properly terminate GLFW
        glfw.terminate()
