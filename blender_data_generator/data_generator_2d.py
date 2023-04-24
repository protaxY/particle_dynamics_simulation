import random
import math
import json
from pathlib import Path

import bpy, bmesh
from mathutils.bvhtree import BVHTree

# number of rollouts
rollout_number = 3000

# output file name
output_directory = Path("dataset")
output_directory.mkdir(parents=True, exist_ok=True)

# smoke cache folder
smoke_cache_folder = Path("cache")
output_directory.mkdir(parents=True, exist_ok=True)

# animation
domain_resolution = 256
start_frame_offset = 30
final_frame_offset = 350
min_animation_frames = 200
max_anumation_frames = 300
bpy.data.scenes["test"].frame_start = 0
bpy.data.scenes["test"].frame_end = 500

max_animated_rotation = 3*math.pi/4

# box
max_box_co = (5, 5, 5, 1)
min_box_co = (-5, -5, -5, 1)

# prop scale
max_prop_scale = 1.2
min_prop_scale = 0.4

# particles number
particles_number = 200
max_gen_attemps_number = 10

floor_scale_offset = 1.5
# желательно, чтобы domain_scale_offset был не меньше, чем floor_scale_offset
domain_scale_offset = 1.5

# допустимые отсутпы от границ расчетной области для частиц
# нарушив их, дальнейшие ее позиции записываться не будут а частица будет отмечена, как не валидная
particles_border_offset = 0.1
initializtion_particles_border_offset = particles_border_offset+0.001

# допустиморе расстояние перемещения частицы до начала передвижения объектов сцены
initial_position_tolerance = 0.05

manual_object_generation = True
object_prop_index = 0
object_scale_x = 1
object_scale_y = 1

#____________________________________________________________________________________________

print("initial preparation...")

bpy.data.objects["smoke_domain"].modifiers["Fluid"].domain_settings.cache_directory = str(smoke_cache_folder)

bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
bpy.context.scene.frame_set(0)

max_non_animated_frame = start_frame_offset

# bounds
bounds = bpy.data.objects["bounds"]
bounds.scale = ((max_box_co[0]-min_box_co[0])*floor_scale_offset,
                (max_box_co[1]-min_box_co[1])*floor_scale_offset, 
                (max_box_co[2]-min_box_co[2])*floor_scale_offset)
bounds.location = ((min_box_co[0]+max_box_co[0])/2,
                   (min_box_co[1]+max_box_co[1])/2,
                   (min_box_co[2]+max_box_co[2])/2)

# prop
objects_on_scene_collection = bpy.data.collections["objects_on_scene"]
rigid_body_world_collection = bpy.data.collections["rigid_body_world"]
bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["objects_on_scene"]

props_collection = bpy.data.collections["props"]
props_number = len(props_collection.objects)

if manual_object_generation:
    current_props_index = object_prop_index
    current_object = bpy.data.objects.new("scene_object", 
                                          props_collection.objects[current_props_index].data.copy())

    current_object.location = (0, 0, 0)
    current_object.scale = (object_scale_x,
                            object_scale_y,
                            1)
else:
    current_props_index = random.randint(0, props_number-1)
    current_object = bpy.data.objects.new("scene_object", 
                                          props_collection.objects[current_props_index].data.copy())

    current_object.location = (0, 0, 0)
    current_object.scale = (random.uniform(min_prop_scale, max_prop_scale),
                            random.uniform(min_prop_scale, max_prop_scale),
                            1)
current_object.rotation_euler[2] = math.pi/2

# apply all transforms
current_object.data.transform(current_object.matrix_basis)
current_object.matrix_basis.identity()

current_object.constraints.new(type='FOLLOW_PATH')
#current_object.constraints[0].use_curve_follow = True
#current_object.constraints[0].target = curves_on_scene_collection.objects["path"]

# particle rasterization
current_object.modifiers.new("scene_object_rasterization", type='PARTICLE_SYSTEM')
bpy.data.particles["ParticleSettings"].name = "scene_object_rasterization"
current_particle_settings = bpy.data.particles["scene_object_rasterization"]
current_particle_settings.frame_start = 0
current_particle_settings.frame_end = 0
current_particle_settings.lifetime = 1
current_particle_settings.emit_from = "VOLUME"
current_particle_settings.distribution = "GRID"
current_particle_settings.grid_resolution = 8
current_particle_settings.render_type = 'OBJECT'
current_particle_settings.instance_object = bpy.data.objects["rigid_particle_prop"]

objects_on_scene_collection.objects.link(current_object)  

bpy.ops.object.select_all(action='DESELECT')
current_object.select_set(True)
bpy.ops.object.duplicates_make_real()
bpy.context.view_layer.objects.active = current_object
bpy.ops.object.parent_set()

bpy.ops.object.select_all(action='DESELECT')
for current_child in current_object.children:
    if current_child.location[1] > 0:
        current_child.select_set(True)
    else:
        current_child.location[1] = 0
        rigid_body_world_collection.objects.link(current_child)   
        current_child.modifiers.new(name='smoke_flow', type='FLUID')
        current_child.modifiers['smoke_flow'].fluid_type = 'FLOW'
        current_child.modifiers['smoke_flow'].flow_settings.flow_behavior = 'INFLOW'
        current_child.modifiers['smoke_flow'].flow_settings.use_initial_velocity = True
        current_child.modifiers['smoke_flow'].flow_settings.temperature = 0
bpy.ops.object.delete()

print("done")

for rollout_id in range(rollout_number):
    print("rollout: ", rollout_id)
    print("scene preparation...")
    
    # rigid body world initialization
    bpy.ops.rigidbody.world_add()
    rigid_body_world_collection = bpy.data.collections["rigid_body_world"]

    bpy.data.scenes["test"].rigidbody_world.collection = rigid_body_world_collection
    bpy.data.scenes["test"].rigidbody_world.constraints = rigid_body_world_collection

    bpy.data.scenes["test"].rigidbody_world.use_split_impulse = True
    bpy.data.scenes["test"].rigidbody_world.point_cache.frame_end = bpy.data.scenes["test"].frame_end
    #bpy.data.scenes["test"].rigidbody_world.time_scale = 0.1
    #bpy.data.scenes["test"].rigidbody_world.substeps_per_frame = 100
    
    bounds.rigid_body.type = 'PASSIVE'
    bounds.rigid_body.collision_shape = 'MESH'
    for current_child in current_object.children:
        current_child.rigid_body.type = 'PASSIVE'
        current_child.rigid_body.collision_shape = 'SPHERE'
    
    # curve
    curves_on_scene_collection = bpy.data.collections["curves_on_scene"]
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["curves_on_scene"]

    bpy.ops.curve.primitive_nurbs_path_add()
    bpy.context.object.name = "path"
        
    current_path = bpy.context.object
        
    path_points_number = random.randint(4, 4)
    current_path.data.splines[0].points.add(path_points_number-5)
    last_point_co = None

    for point_i in range(path_points_number):
        point_co = (random.uniform(min_box_co[0], max_box_co[0]),
                    0,
                    random.uniform(min_box_co[2], max_box_co[2]),
                    1)
        current_path.data.splines[0].points[point_i].co = point_co

    # use generated curve as path to follow
    current_object.constraints[0].target = current_path
    # animation
    animation_start_frame = random.randint(bpy.data.scenes["test"].frame_start+start_frame_offset, 
                                           bpy.data.scenes["test"].frame_end-final_frame_offset)
    max_non_animated_frame = min(max_non_animated_frame, animation_start_frame)

    animation_duration = random.randint(min_animation_frames, max_anumation_frames)
        
    current_object.constraints[0].offset = 0
    current_object.keyframe_insert(data_path='constraints[0].offset', frame=animation_start_frame)
    current_object.constraints[0].offset = -100
    current_object.keyframe_insert(data_path='constraints[0].offset', frame=animation_start_frame+animation_duration)

    # rotate animation
    rotation_angle = random.uniform(0, max_animated_rotation)
    rotation_animation_start_frame = random.randint(bpy.data.scenes["test"].frame_start+start_frame_offset, 
                                                    bpy.data.scenes["test"].frame_end-final_frame_offset)
    max_non_animated_frame = min(max_non_animated_frame, rotation_animation_start_frame)

    rotation_animation_duration = random.randint(min_animation_frames, max_anumation_frames)
        
    current_object.rotation_euler[1] = 0
    current_object.keyframe_insert(data_path='rotation_euler', frame=rotation_animation_start_frame)
    current_object.rotation_euler[1] = rotation_angle
    current_object.keyframe_insert(data_path='rotation_euler', frame=rotation_animation_start_frame+rotation_animation_duration)

    bpy.data.scenes["test"].frame_current = 1
    bpy.data.scenes["test"].frame_current = 0

    particles_on_scene_collection = bpy.data.collections["particles_on_scene"]
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["particles_on_scene"]

    # радиус частицы с запасом
    particle_radius = 0.07

    for particle_i in range(particles_number):
        # частица считается пригодной, если она не пересекается и не находится внутри других объектов.
        is_good = False
        attemps_number = 0
        
        current_particle_object = bpy.data.objects.new("particle_"+str(particle_i), 
                                                       bpy.data.objects["particle_prop"].data)
        current_particle_object.lock_location[1] = True
        particles_on_scene_collection.objects.link(current_particle_object)
        bpy.data.scenes["test"].rigidbody_world.collection.objects.link(current_particle_object)
            
        while not is_good and attemps_number < max_gen_attemps_number:
            attemps_number += 1
            current_particle_object.location = (random.uniform(min_box_co[0]+initializtion_particles_border_offset, max_box_co[0]-initializtion_particles_border_offset),
                                                0,
                                                random.uniform(min_box_co[2]+initializtion_particles_border_offset, max_box_co[2]-initializtion_particles_border_offset))
            
            # предполагаю, что изначально позиция частицы хорошая
            is_good = True
            
            current_particle_bvh_tree = BVHTree.FromObject(current_particle_object, bpy.context.evaluated_depsgraph_get())
            bpy.context.view_layer.update()
            for prop_index, prop_object in enumerate(objects_on_scene_collection.objects):        
                prop_location = None
                prop_location = prop_object.matrix_world @ prop_object.location
                if (current_particle_object.location-prop_location).length < prop_object.scale[0]+particle_radius:
                    is_good = False
                    continue
            for particle_object in particles_on_scene_collection.objects:    
                if particle_object.name != current_particle_object.name and \
                   (current_particle_object.location-particle_object.location).length < particle_radius:
                    is_good = False
                    continue

        if attemps_number == max_gen_attemps_number:
            print("failed to generate particle, attemps limit reached")
            bpy.data.objects.remove(current_particle_object, do_unlink=True)

    # doamin
    domain = bpy.data.objects["smoke_domain"]
    domain.location = (0, 0, (max_box_co[2]+min_box_co[2])/2)

    domain.scale = ((max_box_co[0]-min_box_co[0])*domain_scale_offset,
                    0.2,
                    (max_box_co[2]-min_box_co[2])*domain_scale_offset)

    domain.modifiers[0].domain_settings.cache_frame_start = 0
    domain.modifiers[0].domain_settings.cache_frame_end = bpy.data.scenes["test"].frame_end
    domain.modifiers[0].domain_settings.resolution_max = domain_resolution
    bpy.context.view_layer.objects.active = domain

#    bpy.ops.object.select_all(action='DESELECT')
#    domain.select_set(True)
#    bpy.ops.fluid.free_all()
#    bpy.ops.ptcache.free_bake_all()
    print("done")
    print("particle dynamics calcualtion...")
    bpy.ops.fluid.bake_all()
    bpy.ops.ptcache.bake_all(bake=True)
    print("done")
    
    # удаление частиц, чей импульс в начальный момент времени не равен нулю
    detetion_counter = 0

    # ключ - имя частицы, значение - номер кадра, после которого частица перестает быть валидной
    invalid_particle_names = {}

    bpy.ops.object.select_all(action='DESELECT')
    for current_particle in particles_on_scene_collection.objects:
        bpy.context.scene.frame_set(0)
        start_particle_position = current_particle.matrix_world.translation.copy()
        bpy.context.scene.frame_set(max_non_animated_frame-1)
        first_frame_particle_position = current_particle.matrix_world.translation.copy()
        
        if (start_particle_position-first_frame_particle_position).length > initial_position_tolerance:
            invalid_particle_names[current_particle.name] = 0
            current_particle.select_set(True)
            detetion_counter += 1

    print('deletion_counter: ', detetion_counter, '\n')

#    print(max_non_animated_frame)
#    print(invalid_particle_names)

    # запись rollout-а в json-файл
    print("rollout file writing")
    
    current_rollout_json_path = output_directory / (str(rollout_id)+".json")
    file = open(current_rollout_json_path, "w")

    data = {'end_frame': bpy.data.scenes["test"].frame_end,
            'left_border': bounds.location[0]-bounds.scale[0]/2,
            'right_border': bounds.location[0]+bounds.scale[0]/2,
            'bottom_border': bounds.location[2]-bounds.scale[2]/2,
            'top_border': bounds.location[2]+bounds.scale[2]/2,
            'object_positions': [[]for _ in range(len(current_object.children))], 
            'particle_positions': [[]for _ in range(len(particles_on_scene_collection.objects))]} 
    for current_frame in range(0, bpy.data.scenes["test"].frame_end+1):
        bpy.context.scene.frame_set(current_frame)
        for current_object_particle_id, current_object_particle in enumerate(current_object.children):
            data['object_positions'][current_object_particle_id].append([current_object_particle.matrix_world.translation[0],
                                                                         current_object_particle.matrix_world.translation[2]])
        for current_particle_id, current_particle in enumerate(particles_on_scene_collection.objects):
            if current_particle.matrix_world.translation[0] < bounds.location[0]-bounds.scale[0]/2+particles_border_offset or \
               current_particle.matrix_world.translation[0] > bounds.location[0]+bounds.scale[0]/2-particles_border_offset or \
               current_particle.matrix_world.translation[2] < bounds.location[2]-bounds.scale[2]/2+particles_border_offset or \
               current_particle.matrix_world.translation[2] > bounds.location[2]+bounds.scale[2]/2-particles_border_offset:
                   invalid_particle_names[current_particle.name] = current_frame
                   continue
            if not current_particle.name in invalid_particle_names:
                data['particle_positions'][current_particle_id].append([current_particle.matrix_world.translation[0], 
                                                                        current_particle.matrix_world.translation[2]])
    file.write(json.dumps(data))
    file.close()
    print("done")
    
    # отчистка сцены
    print("cleaning up...")
    
#    if rollout_id != rollout_number-1:
    current_object.animation_data_clear()
    for current_particle_object in particles_on_scene_collection.objects:
        bpy.data.objects.remove(current_particle_object, do_unlink=True)
    bpy.data.objects.remove(current_path, do_unlink=True)
    bpy.ops.outliner.orphans_purge()
    bpy.ops.rigidbody.world_remove()
    print("done")
    
    print("rollout number done\n")
    