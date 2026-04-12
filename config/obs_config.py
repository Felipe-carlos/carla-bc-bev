obs_configs = {
    'hero': {
        'speed': {
            'module': 'actor_state.speed'
        },
        'control': {
            'module': 'actor_state.control'
        },
        'velocity': {
            'module': 'actor_state.velocity'
        },
        'birdview': {
            'module': 'birdview.chauffeurnet',
            'width_in_pixels': 192,
            'pixels_ev_to_bottom': 40,
            'pixels_per_meter': 5.0,
            'history_idx': [-16, -11, -6, -1],
            'scale_bbox': True,
            'scale_mask_col': 1.0
        },
        'route_plan': {
            'module': 'navigation.waypoint_plan',
            'steps': 20
        },
        'gnss': {
            'module': 'navigation.gnss'
        },    
        'central_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,  
            'location': [1.2, 0.0, 1.3],
            'rotation': [0.0, 0.0, 0.0]
        },
        'left_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,  
            'location': [1.2, -0.25, 1.3],
            'rotation': [0.0, 0.0, -45.0]
        },
        'right_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,  
            'location': [1.2, 0.25, 1.3],
            'rotation': [0.0, 0.0, 45.0]
        },
         'rear_rgb': {                      #####added
                'module': 'camera.rgb',
                'fov': 90,
                'width': 256,
                'height': 144,  
                'location': [-1.5, 0.0, 1.3],
                'rotation': [0.0, 0.0, 180.0]
            }
    }
}


