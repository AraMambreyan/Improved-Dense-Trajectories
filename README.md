# Improved Dense Trajectories

Interface for running
[Improved Dense Trajectories](http://lear.inrialpes.fr/~wang/improved_trajectories)
with no extra code. 

The repository allows you to extract Fisher Vectors of HOG/HOF/MBH descriptors for each 
of your videos. Alternatively, you can extract the raw IDT features.

The [original code for IDTs](https://github.com/chuckcho/iDT) uses old versions of OpenCV and ffmpeg,
and installing these is a major headache. A [Python wrapper with a newer version 
of OpenCV](https://github.com/FXPAL/densetrack) was written but it still requires extra code. 
In addition, Fisher Vectors were not implemented in these repositories.

NOTE: I wrote `Dockerfile` and `run_idt.py` files. The rest of the repository is adapted from
the above two repositories.


## Setup

First, if you don't have Docker, [install it](https://docs.docker.com/engine/install/#server).

After installing Docker, clone the repository and move to the cloned directory:

```
git clone https://github.com/AraMambreyan/Improved-Dense-Trajectories.git
cd Improved-Dense-Trajectories
```

Put all your video files in the `data` directory. 

## Modes

The code can be run in two modes:

#### Fisher Vectors

In this mode, a Fisher Vector encoding is applied. For each descriptor (HOG/HOF/MBH), a separate
*csv* file will be outputted in the `features` directory. Each row of the *csv* file is 
the Fisher Vector of one video from the `data` directory.

You can disable the descriptors that you don't want to run by assigning a `False` value to the
accompanying variables at the top of the `run_idt.py` script. For instance, the below will 
only run MBH features. 

```python
HOG_FISHER_VECTOR = False
HOF_FISHER_VECTOR = False
MBH_FISHER_VECTOR = True
```

Note that the final MBH Fisher Vector is obtained by concatenating MBHx and MBHy Fisher Vectors.

#### Raw IDT features

For each video, raw IDT features (trajectories) will be saved with `np.save` (which produces
`.npy` extension) without applying any feature encoding. Even a 30-second video might have 1 million
trajectories so make sure you have enough space. To run in this mode, assign a `False` value
to all accompanying variables at the top of the `run_idt.py` script.

The format of the features are described in [this page](http://lear.inrialpes.fr/~wang/dense_trajectories).
Each trajectory is of `np.void` type and is not very comprehensible. Running the following will
help you understand the format better:

```python
a = np.load('features/<your_video_name_from_data_directory>-trajectory.npy')
print(np.array(a[0]))
```

These two lines simply print the raw features of a single trajectory.

## Running

After the setup and choosing which mode you'd like to run by assigning appropriate boolean variables, you need
to create a Docker image from the `Dockerfile`. Make sure you are in the repository's directory 
and run:

```
docker build -t idt .
```

It would run for a while. Once it finishes, create a Docker container. For **Unix**:

```
docker run -v $(pwd)/features:/densetrack/features idt
```

For **Windows**:

```
docker run -v %cd%\features:/densetrack/features idt
```

I didn't test for Windows so if it doesn't work simply replace `%cd%\features` with the path of the `features`
directory.

That's it! Your features are now in the (wait for it) `features` directory.

#### Clean Up

If you'd like to clean up, run `docker ps --all`, find the container with the image
`idt`, locate its container ID and run `docker container rm <CONTAINER-ID>`. 

To remove the image, run `docker images`, locate the image ID of the `idt` image and run 
`docker rmi -f <IMAGE-ID>`.

## Bugs and Extensions

Feel free to create an issue for bugs or questions. For pull requests, please make sure to test thoroughly.

## Advanced Use

To change the hyperarameters of the IDT, you need to change the `densetrack.densetrack` 
parameters when calling it in the `run_idt` script (under the `main` function). Below are the
default parameters. (I used `adjust_camera=True` when calling the method so that it extracts *Improved*
Dense Trajectories.)

```python
track_length = 15
min_distance = 5
patch_size = 32
nxy_cell = 2
nt_cell = 3
scale_num = 8
init_gap = 1
poly_n = 7
poly_sigma = 1.5
image_pattern = None
adjust_camera = False
```

## Citation

Citation for the paper describing the algorithm:

```
@INPROCEEDINGS{Wang2013,
  author={Heng Wang and Cordelia Schmid},
  title={Action Recognition with Improved Trajectories},
  booktitle= {IEEE International Conference on Computer Vision},
  year={2013},
  address={Sydney, Australia},
  url={http://hal.inria.fr/hal-00873267}
}
```
