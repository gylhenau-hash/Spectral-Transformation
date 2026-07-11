#%%
import numpy as np
import tensorflow as tf
import rasterio

# define input and output
input_tif = r"s2toa Dianchi.tif"
output_tif = r"embedding_20d_Dianchi.tif"
path_model = r"student_s2.keras"
path_mean = r"mean_s2.npy"
path_std = r"std_s2.npy"

# read model
encoder = tf.keras.models.load_model(path_model,safe_mode=False)

# read normalization parameters
mean = np.load(path_mean)
std = np.load(path_std)

# read image
with rasterio.open(input_tif) as src:
    img = src.read()          
    profile = src.profile
img = np.transpose(
    img,
    (1,2,0)
)                             
H,W,B = img.shape
print(img.shape)

# flatten image
X = img.reshape(-1, B).astype(np.float32)
print(X.shape)

# standardization
X = (X - mean) / std

# detect invalid pixels
valid_mask = np.all(
    np.isfinite(X),
    axis=1
)

X_valid = X[valid_mask]
print(X_valid.shape)

# predict
embedding_valid = encoder.predict(
    X_valid,
    batch_size=4096,
    verbose=1
)

# create embedding
embedding = np.full(
    (X.shape[0],20),
    np.nan,
    dtype=np.float32
)

embedding[valid_mask] = embedding_valid

# reshape
embedding_img = embedding.reshape( H, W, 20)
print(embedding_img.shape)

# transpose
embedding_img = np.transpose(
    embedding_img,
    (2,0,1)
)

# update profile
profile.update(
    dtype=rasterio.float32,
    count=20,
    compress='lzw'
)

# write
with rasterio.open(
    output_tif,
    "w",
    **profile
) as dst:
    dst.write(
        embedding_img.astype(np.float32)
    )
