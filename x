unique_tracks = sorted(df['truthID'].unique())

for track in unique_tracks:
    track_data = df[df['truthID'] == track].sort_values('time_s')
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Time-Series Analysis - Track {track} by Object Type', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (col, label) in enumerate(measurements):
        # Plot each object type in this track
        for obj_type in object_types:
            obj_track_data = track_data[track_data['objectType'] == obj_type]
            if len(obj_track_data) > 0:
                color = colors_dict.get(obj_type, 'gray')
                axes[idx].scatter(obj_track_data['time_s'], obj_track_data[col], 
                                alpha=0.6, s=30, color=color, label=obj_type)
        
        axes[idx].set_xlabel('Time (s)', fontsize=11)
        axes[idx].set_ylabel(label, fontsize=11)
        axes[idx].set_title(f'{label} over Time', fontsize=12, fontweight='bold')
        axes[idx].legend(loc='best')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'timeseries_track_{track}_by_objecttype.png', dpi=300, bbox_inches='tight')
    plt.show()




stats_by_type = []

for obj_type in object_types:
    obj_data = df[df['objectType'] == obj_type]
    
    stats = {
        'Object Type': obj_type,
        'Num Detections': len(obj_data),
        'Mean Range (m)': obj_data['meas_range_m'].mean(),
        'Mean SNR (dB)': obj_data['snrInst_dB'].mean(),
        'SNR Std (dB)': obj_data['snrInst_dB'].std(),
        'Mean RCS (dBsm)': obj_data['rcsInst_dBsm'].mean(),
        'RCS Std (dBsm)': obj_data['rcsInst_dBsm'].std(),
        'Mean Speed (m/s)': obj_data['meas_rr_mps'].mean(),
        'Mean Azimuth (deg)': obj_data['meas_az_deg'].mean(),
        'Mean Elevation (deg)': obj_data['meas_el_deg'].mean()
    }
    stats_by_type.append(stats)

stats_df = pd.DataFrame(stats_by_type)
print("\nSummary Statistics by Object Type:")
print(stats_df.to_string(index=False))
