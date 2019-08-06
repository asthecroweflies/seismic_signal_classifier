# inspired by https://bspeice.github.io/audio-compression-using-pca.html
import IPython
from IPython.display import Audio
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import numpy as np
from obspy import read, Trace
from matplotlib import pyplot as plt 
from yellowbrick.features.pca import PCADecomposition

block_to_component_dict = {64 : 64, 128 : 90, 256 : 45, 512 : 23, 1024: 12, 2048: 6}

TRACE_SIZE = 2840
N_COMPONENTS = 63
DEFAULT_BLOCK_SIZE = 64
MAX_N_COMPONENTS = block_to_component_dict.get(DEFAULT_BLOCK_SIZE)
playback_sampling_rate = 1000

def plot_wiggle(wiggle):
    num_points = 0
    
    if (len(wiggle.shape) > 1): # 2D
        for a in range(0, len(wiggle)):
            for b in range(0, len(wiggle[a])):
                num_points += 1
        #print("point sum: " + str(num_points))
        x = np.arange(0, num_points) 
        y = wiggle.flatten()
        plt.imshow(wiggle)
        plt.show()
    else:                       # 1D
        for a in range(0, len(wiggle)):
            num_points += 1
        x = np.arange(0, num_points)
        y = wiggle

    #plt.plot(y, color="#c4000f", linewidth=0.8)
    plt.plot(y, color="#c71421", linewidth=1.7) 
    plt.show()
    
def normalize1D(data):
    #z = xi - min(x) / (max(x) - min(x))
    return data/np.linalg.norm(data, ord=2, axis=0, keepdims=True)

def project_pca(X):
    
    #colors = np.array(['r' if yi else 'b' for yi in y])
    vis = PCADecomposition(scale=True, proj_features=True, proj_dim=3)#, color=colors)
    vis.fit_transform(X)
    vis.poof()

def pca_reduce(trace_data, n_components, block_size):
    # standardize data stream (trace lengths are variable)
    wiggle = trace_data#trace.data#[:TRACE_SIZE] 
    normalized_wiggle = normalize1D(wiggle)

    # padding wiggle to make it divisible by block_size
    samples = len(wiggle)#trace.stats.npts
    #print("\nsamples length " + str(samples))
    hanging = block_size - np.mod(samples, block_size)
    #print("hanging : " + str(hanging))

    last_data_pt = normalized_wiggle[len(normalized_wiggle) - 1]
    padded  = np.lib.pad(normalized_wiggle, (0, hanging), 'constant', constant_values=last_data_pt)

    # reshape wiggle to have block-size dimensions
    reshaped_wiggle = padded.reshape((len(padded) // block_size, block_size))
    # Perform principal component analysis / dimensionality reduction

    min_n_components = min(n_components, block_size)
    pca = PCA(n_components=min_n_components, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)

    # find optimal projection plane
    pca.fit(reshaped_wiggle)

    # actually project data onto plane
    transformed_wiggle = pca.transform(reshaped_wiggle)
    
    # this shows us how much data was lost / deemed irrelevant 
    reconstructed_wiggle = pca.inverse_transform(transformed_wiggle).reshape((len(padded)))
    
    return pca, transformed_wiggle, reconstructed_wiggle

# if using Jupyter notebook this function will allow you to hear & see the original vs reconstructed wiggles
def pca_display(pca, original_trace, transformed_wiggle, reconstructed_wiggle):
    original_wiggle = original_trace.data
    IPython.display.display(IPython.display.Audio(data=original_wiggle, rate=playback_sampling_rate))
    IPython.display.display(IPython.display.Audio(data=reconstructed_wiggle, rate=playback_sampling_rate))
    
    plot_wiggle(transformed_wiggle)

    reduced_trace = Trace(data=reconstructed_wiggle, header=None)
    
    print("principal_component  length: "  + str(len(reduced_trace)))
    print("PCA Variance Ratio: ")
    print(pca.explained_variance_ratio_)
    
    original_trace.plot()
    reduced_trace.plot()

def main():
    stream_path_drilling = "D:\\labeled_data\\labeled_triggers\\drilling\\1544195969.55-l.mseed"
    stream_path_ert      =  "D:\\labeled_data\\labeled_triggers\\ert\\1544102595.07-l.mseed"
    st                   = read(stream_path_ert)
    ob16                 = st[54]
    print(ob16.stats)
    print("ob16.data length: "  + str(len(ob16.data)))

    try:
        pca, transformed_wiggle, reconstructed = pca_reduce(ob16.data, N_COMPONENTS, DEFAULT_BLOCK_SIZE)
        pca_display(pca, ob16, transformed_wiggle, reconstructed)

    except ValueError:
        print("")

    
if __name__ == "__main__":
    main()

