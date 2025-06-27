import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import os
import numpy as np
import seaborn as sns

df = pd.read_csv('vv2/2023-05-15_sm00_udp1_#01_All.csv')



root = "./vv2/"

ho_types = ['LTE_HO', 'MN_HO', 'SN_HO','without_HO']
aggregated_pre          = {ho: [] for ho in ho_types}
aggregated_post         = {ho: [] for ho in ho_types}
aggregated_pre_latency  = {ho: [] for ho in ho_types}
aggregated_post_latency = {ho: [] for ho in ho_types}





#directory = "2024-06-18-1/UDP_Bandlock_9S_Phone_B/sm02/#01/"
files = []
for file in os.listdir(root):
    if file == "record.csv":
        continue
    if not file.lower().endswith('.csv'):
        continue

    file_path = os.path.join(root, file)
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')
    files.append(file)

    print("file:", file)

    if 'Timestamp' not in df.columns:
        continue


    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    df['HO_Flag']     = (df[['LTE_HO','MN_HO','SN_HO']] > 0).any(axis=1).astype(int)
    df['LTE_HO_Flag'] = (df['LTE_HO'] > 0).astype(int)
    df['MN_HO_Flag']  = (df['MN_HO']  > 0).astype(int)
    df['SN_HO_Flag']  = (df['SN_HO']  > 0).astype(int)

    window = pd.Timedelta(seconds=5)


    for ho in ho_types:
        if ho == 'without_HO':
            # 沒有任何 HO 發生的時間點
            ho_times = df.loc[df['HO_Flag'] == 0, 'Timestamp']
        else:
            ho_times = df.loc[df[ho] > 0, 'Timestamp']

        if ho_times.empty:
            continue

        sample_times = ho_times.iloc[:1000]
        dl_pre, dl_post = [], []
        lat_pre, lat_post = [], []

        for t0 in sample_times:
            pre_mask  = (df['Timestamp'] >= t0 - window) & (df['Timestamp'] <  t0)
            post_mask = (df['Timestamp'] >  t0) & (df['Timestamp'] <= t0 + window)

            dl_pre.append( df.loc[pre_mask,  'dl-loss'].values )
            dl_post.append(df.loc[post_mask, 'dl-loss'].values )
            lat_pre.append(df.loc[pre_mask,  'dl-latency'].values )
            lat_post.append(df.loc[post_mask, 'dl-latency'].values )

        aggregated_pre[ho].extend(np.concatenate(dl_pre))       if dl_pre       else None
        aggregated_post[ho].extend(np.concatenate(dl_post))     if dl_post      else None
        aggregated_pre_latency[ho].extend(np.concatenate(lat_pre))   if lat_pre   else None
        aggregated_post_latency[ho].extend(np.concatenate(lat_post)) if lat_post  else None
            




pctles = np.arange(0, 101, 10)
for ho in ho_types:
    pre = np.array(aggregated_pre[ho])
    post = np.array(aggregated_post[ho])
    print(f"===== {ho} =====")
    print('Pre 5s Mean:', np.mean(pre), 'Post 5s Mean:', np.mean(post))
    print('Pre 5s Median:', np.median(pre), 'Post 5s Median:', np.median(post))
    print('Pre 5s P90:', np.percentile(pre, 90), 'Post 5s P90:', np.percentile(post, 90))
    # CDF
    pre_vals = np.percentile(pre, pctles)
    post_vals = np.percentile(post, pctles)
    cdf_df = pd.DataFrame({
        'Percentile': pctles,
        f'{ho}_Pre_CDF': pre_vals,
        f'{ho}_Post_CDF': post_vals
    })
    print(cdf_df)

    lat_pre = np.array(aggregated_pre_latency[ho])
    lat_post = np.array(aggregated_post_latency[ho])
    print('Pre 5s Latency Mean:', np.mean(lat_pre), 'Post 5s Latency Mean:', np.mean(lat_post))
    print('Pre 5s Latency Median:', np.median(lat_pre), 'Post 5s Latency Median:', np.median(lat_post))
    print('Pre 5s Latency P90:', np.percentile(lat_pre,90), 'Post 5s Latency P90:', np.percentile(lat_post,90))
    # CDF
    lat_pre_vals = np.percentile(lat_pre, pctles)
    lat_post_vals = np.percentile(lat_post, pctles)
    lat_cdf_df = pd.DataFrame({
        'Percentile': pctles,
        f'{ho}_Pre_Lat_CDF': lat_pre_vals,
        f'{ho}_Post_Lat_CDF': lat_post_vals
    })
    print(lat_cdf_df)





#print(files)



stats_loss = pd.DataFrame({
    'Pre Mean Loss': [np.mean(aggregated_pre[ho]) for ho in ho_types],
    'Post Mean Loss': [np.mean(aggregated_post[ho]) for ho in ho_types]
}, index=ho_types)
# Loss
stats_loss.plot(kind='bar')
plt.title('Mean DL Loss Pre vs Post 5s by HO Type')
plt.xlabel('HO Type')
plt.ylabel('Mean DL Loss')
plt.tight_layout()
plt.show()

# Loss Pre CDF
plt.figure()
for ho in ho_types:
    pre_vals = np.percentile(aggregated_pre[ho], pctles)
    plt.plot(pctles, pre_vals, label=ho)
plt.title('DL Loss Pre CDF by HO Type')
plt.xlabel('Percentile')
plt.ylabel('DL Loss Rate')
plt.legend()
plt.tight_layout()
plt.show()

#  Loss Post CDF
plt.figure()
for ho in ho_types:
    post_vals = np.percentile(aggregated_post[ho], pctles)
    plt.plot(pctles, post_vals, label=ho)
plt.title('DL Loss Post CDF by HO Type')
plt.xlabel('Percentile')
plt.ylabel('DL Loss Rate')
plt.legend()
plt.tight_layout()
plt.show()

# DataFrame
stats_latency = pd.DataFrame({
    'Pre Mean Latency': [np.mean(aggregated_pre_latency[ho]) for ho in ho_types],
    'Post Mean Latency': [np.mean(aggregated_post_latency[ho]) for ho in ho_types]
}, index=ho_types)
# Latency
stats_latency.plot(kind='bar')
plt.title('Mean DL Latency Pre vs Post 5s by HO Type')
plt.xlabel('HO Type')
plt.ylabel('Mean DL Latency (ms)')
plt.tight_layout()
plt.show()

# Latency Pre CDF
plt.figure()
for ho in ho_types:
    lat_pre_vals = np.percentile(aggregated_pre_latency[ho], pctles)
    plt.plot(lat_pre_vals,pctles, label=ho)
plt.title('DL Latency Pre CDF by HO Type')
plt.ylabel('Percentile')
plt.xlabel('DL Latency (ms)')
plt.legend()
plt.tight_layout()
plt.show()

# Latency Post CDF
plt.figure()
for ho in ho_types:
    lat_post_vals = np.percentile(aggregated_post_latency[ho], pctles)
    plt.plot(lat_post_vals,pctles, label=ho)
plt.title('DL Latency Post CDF by HO Type')
plt.ylabel('Percentile')
plt.xlabel('DL Latency (ms)')
plt.legend()
plt.tight_layout()
plt.show()
