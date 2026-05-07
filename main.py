import pandas as pd
import seaborn as sns
import time
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from my_hdbscan import HDBSCAN

if __name__ == '__main__':
    plot_kwds = {'alpha': 0.5, 's': 50, 'linewidth': 0}

    df = pd.read_csv('Rodent_Inspection_20251121.csv')
    pd.set_option('display.max_columns', None)

    df.dropna(inplace=True)
    df = df.head(n=1000000)

    X = df[['LONGITUDE', 'LATITUDE', 'INSPECTION_TYPE']]

    X = X.reset_index(drop=True)

    numeric_features = ['LONGITUDE', 'LATITUDE']
    categorical_features = ['INSPECTION_TYPE']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    X_scaled = preprocessor.fit_transform(X)

    print("HDBSCAN Beginning!")

    start_time = time.perf_counter()
    cpu_start = time.process_time()

    hdb = HDBSCAN(min_samples=20, min_cluster_size=20)
    hdb.fit(X_scaled)

    end_time = time.perf_counter()
    cpu_end = time.process_time()

    total_time = end_time - start_time
    print(f"HDBSCAN Runtime: {total_time:.3f}")

    total_cpu = cpu_end - cpu_start
    print(f"HDBSCAN CPU Time: {total_cpu:.3f}")

    print(f"Number of clusters: {len(set(hdb.labels_)) - 1}")

    X_displayed = pd.DataFrame(X_scaled)

    unique_styles = X['INSPECTION_TYPE'].unique()

    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    style_map = dict(zip(unique_styles, markers[:len(unique_styles)]))

    ax = sns.scatterplot(
        x=X_displayed[0],
        y=X_displayed[1],
        hue=hdb.labels_,
        style=X['INSPECTION_TYPE'],
        markers=style_map,
        palette='deep',
        legend=False,
        **plot_kwds
    )

    handles = [
        Line2D([0], [0], marker=style_map[style], color='black',
               linestyle='None', markersize=8, label=style)
        for style in unique_styles
    ]

    ax.legend(handles=handles, title="Inspection Type", loc='best')

    plt.show()
