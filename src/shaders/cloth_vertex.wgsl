#import bevy_pbr::{
    mesh_functions,
    mesh_bindings::mesh,
    view_transformations::position_world_to_clip,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::{Vertex, VertexOutput};
#else
#import bevy_pbr::forward_io::{Vertex, VertexOutput};
#endif

#ifndef PREPASS_PIPELINE
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
}
#import bevy_pbr::forward_io::FragmentOutput
#endif

@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<storage, read> sim_positions: array<vec4<f32>>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<storage, read> sim_normals: array<vec4<f32>>;

@vertex
fn vertex(vertex: Vertex, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
#ifdef MORPH_TARGETS
    return out;
#endif
    // Particle id: prefer UV1 — matches rest mesh (`uv_b.x = particle index`) when `VERTEX_UVS_B` is on.
    // Fallback: indexed draw identity (`vertex_index` + Bevy slab base), same as `draw_indexed` base vertex.
#ifdef VERTEX_UVS_B
    let vi = u32(vertex.uv_b.x + 0.5);
#else
    let vi = vertex_index - mesh[vertex.instance_index].first_vertex_index;
#endif

    let mesh_world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
#ifdef SKINNED
    var world_from_local = bevy_pbr::skinning::skin_model(
        vertex.joint_indices,
        vertex.joint_weights,
        vertex.instance_index
    );
#else
    var world_from_local = mesh_world_from_local;
#endif

#ifndef PREPASS_PIPELINE
#ifdef VERTEX_NORMALS
#ifdef SKINNED
    out.world_normal = bevy_pbr::skinning::skin_normals(world_from_local, vertex.normal);
#else
    let sn = sim_normals[vi].xyz;
    out.world_normal = mesh_functions::mesh_normal_local_to_world(sn, vertex.instance_index);
#endif
#endif
#endif

#ifdef VERTEX_POSITIONS
    let sp = sim_positions[vi].xyz;
    let world =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(sp, 1.0));
#ifdef PREPASS_PIPELINE
    out.world_position = world;
    out.position = position_world_to_clip(world.xyz);
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
#ifdef VERTEX_NORMALS
#ifdef SKINNED
    out.world_normal = bevy_pbr::skinning::skin_normals(world_from_local, vertex.normal);
#else
    let sn = sim_normals[vi].xyz;
    out.world_normal = mesh_functions::mesh_normal_local_to_world(sn, vertex.instance_index);
#endif
#endif
#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        vertex.tangent,
        vertex.instance_index,
    );
#endif
#endif
#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
#else
    out.world_position = world;
    out.position = position_world_to_clip(world.xyz);
#endif
#endif

#ifndef PREPASS_PIPELINE
#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif
#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        vertex.tangent,
        vertex.instance_index,
    );
#endif
#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex.instance_index,
        mesh_world_from_local[3],
    );
#endif
#endif

    return out;
}

#ifndef PREPASS_PIPELINE
@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    // `StandardMaterial.double_sided` + `cull_mode: None` in Rust; back-face albedo below.
    var alb = pbr_input.material.base_color;
    if !is_front {
        // Linear-ish red complement to the demo’s blue front (`examples/cloth_xpbd.rs`).
        alb = vec4<f32>(0.9, 0.06, 0.055, alb.a);
    }
    pbr_input.material.base_color = alpha_discard(pbr_input.material, alb);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
#endif
