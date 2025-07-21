import numpy as np
def sample_frame_indices(n_frames_to_sample, stride, n_total_frames, n_views=1):
    '''
    Sample a given number of frame indices from the video.
    Args:
        n_frames_to_sample (`int`): Total number of frames to sample.
        stride (`int`): Sample every n-th frame.
        n_total_frames (`int`): Maximum allowed index of sample's last frame.
        n_views (`int`): Number of views to sample from the video
    Returns:
        indices (`List[List[int]]`): n_views x List of sampled frame indices for each view
    '''
    if n_total_frames==0:
        raise Exception("no frame in the video")
    
    indices = []

    converted_len = int(n_frames_to_sample * stride) #C:64
    multi_view_converted_len = converted_len*n_views #C: 64
    # max: 4 views, 2 strides, 32 frames = 4*2*32 = 256 frames = ~ 17 sec
    # avg: 4 views, 1 stride, 32 frames = 4*1*32 = 128 frames = ~ 8.5 sec

    if multi_view_converted_len<=n_total_frames:
        # we can sample all multi-views
        # if we have some extra frames, we can start from a random index, and consecutively sample multi views
        end_idx = np.random.randint(multi_view_converted_len, n_total_frames+1)

        start_idx = end_idx - multi_view_converted_len

        while start_idx<end_idx:
            # sample n frames uniformly out of converted_len frames, starting from start_idx 
            local_end_index = start_idx+converted_len
            view_indices = np.linspace(start_idx, local_end_index, num=n_frames_to_sample+1)[:-1].astype(np.int64)
            indices.append(view_indices)
            start_idx = local_end_index
    
    else:
        # we have less frames than required. So, overlapping would be necessary
        # minimum number of frames required for a single view (worst case): 1 view x 2 stride x 32 frames = 64 frames = ~4.3 sec
        if n_total_frames<converted_len:
            raise Exception("do not have enough frames for a single view")
        else:
            end_idx = n_total_frames
            start_idx = 0

            while len(indices)<n_views:
                local_end_index = start_idx+converted_len
                view_indices = np.linspace(start_idx, local_end_index, num=n_frames_to_sample+1)[:-1].astype(np.int64)
                indices.append(view_indices)

                if (local_end_index+converted_len)<n_total_frames:
                    start_idx = local_end_index
                else:
                    start_idx = np.random.randint(start_idx, (n_total_frames-converted_len)+1)
            # end of view collection
            
    return indices

if __name__ == "__main__":
    n_frames_to_sample = 32
    stride = 2
    n_total_frames = [0, 30, 33, 60, 64, 70, 105, 128, 240, 256, 262]
    n_views = 4

    for n_frames in n_total_frames:
        print(f"Number of frames to sample: {n_frames_to_sample}, Stride: {stride}, Available frames: {n_frames}, Views: {n_views}")
        try:
            print(sample_frame_indices(n_frames_to_sample, stride, n_frames, n_views))
        except Exception as e:
            print(e)