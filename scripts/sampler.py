import pandas as pd
from sklearn.utils import resample

def balance_classes(df: pd.DataFrame, col_name: str, n_samples: int) -> pd.DataFrame:
    # Separate classes
    df_go_short = df[df[f'{col_name}'] == 'go_short']
    df_go_long = df[df[f'{col_name}'] == 'go_long']
    df_do_nothing = df[df[f'{col_name}'] == 'do_nothing']


    # 1. Downsample 'go_short' to 980 samples
    '''
    df_go_short_downsampled = resample(df_go_short, 
                                       replace=False,  # No replacement for downsampling
                                       n_samples=n_samples, 
                                       random_state=42)

    df_go_short_downsampled = resample(df_go_short, 
                                       replace=False,  # No replacement for downsampling
                                       n_samples=n_samples, 
                                       random_state=42)
    '''
    # 2. Upsample 'go_long' to 980 samples
    df_go_short_upsampled = resample(df_go_short, 
                                       replace=True,   # Replacement for upsampling
                                       n_samples=n_samples, 
                                       random_state=42)


    # 3. Upsample 'do_nothing' to 980 samples
    df_go_long_upsampled = resample(df_go_long, 
                                       replace=True,   # Replacement for upsampling
                                       n_samples=n_samples, 
                                       random_state=42)
    # 2. Upsample 'go_long' to 980 samples
    '''
    df_go_long_upsampled = resample(df_go_long, 
                                       replace=True,   # Replacement for upsampling
                                       n_samples=n_samples, 
                                       random_state=42)
    '''
    df_do_nothing_upsampled = resample(df_do_nothing, 
                                       replace=True,   # Replacement for upsampling
                                       n_samples=n_samples, 
                                       random_state=42)

    # 3. Combine the balanced classes
    df_balanced = pd.concat([df_go_short_upsampled, df_go_long_upsampled, df_do_nothing])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced

# Test the function
df = pd.read_csv('../jupyter/spreadsheets/rlhf_large_99rl.csv')
df = balance_classes(df, 'action', 294)

print(df['action'].value_counts())

df.to_csv('../jupyter/spreadsheets/rlhf_large_99rl_balanced.csv', index=False)