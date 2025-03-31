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
                "box": "wood.png",
                "pushable_box": "wood.png",
                "linked_door": "wood.png",
                "pressure_plate": "metal.png",
                "lever": "metal.png",
            }.items()
        }
        if "pressure_plate" not in self.textures:
            self.textures["pressure_plate"] = self.textures.get("wall")
        if "lever" not in self.textures:
            self.textures["lever"] = self.textures.get("wall")

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
                # Calculate key dimensions
                handle_outer_radius = 0.15
                handle_inner_radius = 0.05
                stem_length = 0.3
                stem_radius = 0.04
                tooth_width = 0.08
                tooth_height = 0.12
                tooth_depth = 0.06

                # Calculate total key length to center it in the grid cell
                total_key_length = handle_outer_radius * 2 + stem_length
                offset_for_centering = total_key_length / 2

                # Position the handle on the left side
                handle_center_x = (
                    key.pos[0] - offset_for_centering + handle_outer_radius
                )

                # Draw key handle using torus primitive (donut shape)
                glPushMatrix()
                glTranslatef(handle_center_x, -0.25, key.pos[1])
                glRotatef(90, 0, 0, 1)  # Rotate to orient torus in horizontal plane
                render_torus(
                    handle_inner_radius,
                    handle_outer_radius,
                    nsides=16,
                    rings=16,
                    texture=self.textures["key"],
                )
                glPopMatrix()

                # Draw key stem (cylinder primitive)
                # Position the stem to start at the right edge of the donut
                stem_start_x = handle_center_x + handle_outer_radius

                glPushMatrix()
                # Position at the start of the stem
                glTranslatef(stem_start_x, -0.25, key.pos[1])
                # Move the stem length forward since cylinder is centered at origin
                glTranslatef(stem_length, 0, 0)
                glRotatef(90, 0, 0, 1)  # Rotate so that the cylinder lies horizontally
                render_cylinder(
                    0, 0, 0, stem_radius, stem_length, texture=self.textures["key"]
                )
                glPopMatrix()

                # Draw single downward tooth at the end of the stem
                tooth_x = stem_start_x + stem_length - tooth_width / 2

                glPushMatrix()
                glTranslatef(tooth_x, -0.25, key.pos[1])
                glTranslatef(
                    0, -tooth_height / 2, 0
                )  # Move down to create downward tooth
                glScalef(tooth_width, tooth_height, tooth_depth)
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
                # Skip pushable boxes - they'll be rendered separately
                if box.__class__.__name__ != "PushableBox":
                    self._render_box(box.pos[0], box.pos[1])
                else:
                    self._render_pushable_box(box.pos[0], box.pos[1])

        # Render pushable boxes
        if "pushable_boxes" in env.objects:
            for box in env.objects["pushable_boxes"]:
                self._render_pushable_box(box.pos[0], box.pos[1])

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

        # Render Linked Doors
        if "linked_doors" in env.objects:
            for door in env.objects["linked_doors"]:
                if not door.is_open:
                    glPushMatrix()
                    glTranslatef(door.pos[0], 0.0, door.pos[1])
                    render_cube(0, 0.0, 0, self.textures["linked_door"])
                    glPopMatrix()

        # Render Pressure Plates
        if "pressure_plates" in env.objects:
            for plate in env.objects["pressure_plates"]:
                glPushMatrix()
                glTranslatef(plate.pos[0], -0.48, plate.pos[1])
                glScalef(0.8, 0.04, 0.8)
                render_cube(0, 0, 0, self.textures["pressure_plate"])
                glPopMatrix()

        # Render Levers
        if "levers" in env.objects:
            for lever in env.objects["levers"]:
                self._render_lever(lever.pos[0], lever.pos[1], lever.activated)

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
        """Render a box as a chest with a lid."""
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

        # Render a small keyhole on the front
        glPushMatrix()
        glTranslatef(x, -0.1, z + 0.4)  # Front of the box
        glScalef(0.1, 0.1, 0.05)  # Small keyhole
        render_cube(0, 0, 0, None, color=(0.2, 0.2, 0.2))  # Dark color for keyhole
        glPopMatrix()

    def _render_pushable_box(self, x, z):
        """Render a pushable box as a simple cube with directional arrows."""
        # Render main box (simpler than a chest - just a cube)
        glPushMatrix()
        glTranslatef(x, -0.25, z)  # Slightly raised from ground
        glScalef(0.8, 0.6, 0.8)  # Slightly taller than a regular box
        render_cube(0, 0, 0, self.textures["pushable_box"])
        glPopMatrix()

        # Render directional arrows on each side
        arrow_size = 0.2
        arrow_height = 0.1

        # North arrow
        glPushMatrix()
        glTranslatef(x, -0.1, z - 0.4)  # North side
        glRotatef(90, 1, 0, 0)  # Rotate to face north
        render_cylinder(0, 0, 0, 0.05, arrow_size, texture=None, color=(0.3, 0.3, 0.3))
        glPopMatrix()

        # South arrow
        glPushMatrix()
        glTranslatef(x, -0.1, z + 0.4)  # South side
        glRotatef(-90, 1, 0, 0)  # Rotate to face south
        render_cylinder(0, 0, 0, 0.05, arrow_size, texture=None, color=(0.3, 0.3, 0.3))
        glPopMatrix()

        # East arrow
        glPushMatrix()
        glTranslatef(x + 0.4, -0.1, z)  # East side
        glRotatef(90, 0, 0, 1)  # Rotate to face east
        render_cylinder(0, 0, 0, 0.05, arrow_size, texture=None, color=(0.3, 0.3, 0.3))
        glPopMatrix()

        # West arrow
        glPushMatrix()
        glTranslatef(x - 0.4, -0.1, z)  # West side
        glRotatef(-90, 0, 0, 1)  # Rotate to face west
        render_cylinder(0, 0, 0, 0.05, arrow_size, texture=None, color=(0.3, 0.3, 0.3))
        glPopMatrix()

    def _render_lever(self, x, z, activated):
        """Render a lever with a base and handle indicating state."""
        base_color = (0.4, 0.4, 0.4)
        handle_color_active = (0.1, 0.7, 0.1)
        handle_color_inactive = (0.7, 0.1, 0.1)

        glPushMatrix()
        glTranslatef(x, -0.4, z)
        glScalef(0.2, 0.2, 0.2)
        render_cube(0, 0, 0, texture=None, color=base_color)
        glPopMatrix()

        handle_length = 0.4
        handle_radius = 0.05
        handle_color = handle_color_active if activated else handle_color_inactive
        angle = 45 if activated else -45

        glPushMatrix()
        glTranslatef(x, -0.3, z)
        glRotatef(angle, 0, 0, 1)
        glTranslatef(handle_length / 2, 0, 0)
        render_cylinder(
            0, 0, 0, handle_radius, handle_length, texture=None, color=handle_color
        )
        glPopMatrix()

    def render_frame(self, env, agent_idx=0, is_state_view=False):
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._current_agent_idx = agent_idx
        agent = env.agents[agent_idx]
        self.set_camera(agent.pos, agent.looking)

        should_regenerate_list = False
        if "scene" not in self.display_lists or self.last_objects is None:
            should_regenerate_list = True
        else:
            try:
                if any(
                    len(env.objects.get(k, [])) != len(self.last_objects.get(k, []))
                    for k in set(env.objects) | set(self.last_objects)
                ):
                    should_regenerate_list = True
                else:
                    for k in env.objects:
                        if k == "linked_doors":
                            if any(
                                o1.is_open != o2.is_open
                                for o1, o2 in zip(env.objects[k], self.last_objects[k])
                            ):
                                should_regenerate_list = True
                                break
                        elif k == "levers":
                            if any(
                                o1.activated != o2.activated
                                for o1, o2 in zip(env.objects[k], self.last_objects[k])
                            ):
                                should_regenerate_list = True
                                break
            except Exception:
                should_regenerate_list = True

        if should_regenerate_list:
            if "scene" in self.display_lists:
                glDeleteLists(self.display_lists["scene"], 1)
            glfw.make_context_current(self.window)
            self.display_lists["scene"] = self.create_scene_display_list(env)
            self.last_objects = copy.deepcopy(env.objects)

        glCallList(self.display_lists["scene"])

        # Render floor
        render_plane(
            env.grid_width / 2 - 0.5,
            -0.5,
            env.grid_height / 2 - 0.5,
            max(env.grid_width, env.grid_height),
            self.textures["floor"],
            repeat=max(env.grid_width, env.grid_height) / 4,
        )

        # Read pixels
        buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        image = np.flip(image, axis=0)
        image = np.flip(image, axis=1)
        return resize_obs(image, self.resolution, self.torch_obs)

    def render(self, env, agent_idx=0, is_state_view=False):
        # Implement the common interface render method
        return self.render_frame(env, agent_idx, is_state_view)

    def close(self):
        glfw.make_context_current(self.window)
        for list_id in self.display_lists.values():
            glDeleteLists(list_id, 1)
        glfw.destroy_window(self.window)

        # Properly terminate GLFW
        glfw.terminate()
