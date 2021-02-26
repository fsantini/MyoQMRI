[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4279445.svg)](https://doi.org/10.5281/zenodo.4279445)

# MyoQMRI
This project is an open-source effort to put together tools for quantitative MRI of the muscles. Currently, it supports fast water T2 mapping from multi echo spin echo images.

## Citing

If you find this work useful, please cite the following paper:

Santini F, Deligianni X, Paoletti M, et al., *Fast open-source toolkit for water T2 mapping in the presence of fat from multi-echo spin-echo acquisitions for muscle MRI*, Frontiers in Neurology 2021, https://doi.org/10.3389/fneur.2021.630387.

## Multi-Echo Spin Echo

The main function is used to obtain **water T2** and **fat fraction** from
 multi-echo spin echo images as described in [Marty et al. ](https://doi.org/10.1002/nbm.3459)

This implementation uses the GPU for the generation of a dictionary of 
signals through [Extended Phase Graph simulation](https://doi.org/10.1002/jmri.24619).

### Usage

    usage: epgFit.py [-h] [--fit-type T] [--fat-t2 T2] [--noise-level N] [--nthreads T] [--plot-level L] [--t2-limits min max] [--b1-limits min max] [--use-gpu] [--ff-map dir]
                     [--register-ff] [--etl-limit N] [--out-suffix ext] [--slice-range start end] [--refocusing-width factor] [--exc-profile path] [--ref-profile path]
                     path
    
    Fit a multiecho dataset
    
    positional arguments:
      path                  path to the dataset
    
    optional arguments:
      -h, --help            show this help message and exit
      --fit-type T, -y T    type of fitting: T=0: EPG, T=1: Single exponential, T=2: Double exponential (default: 0)
      --fat-t2 T2, -f T2    fat T2 (default: 151)
      --noise-level N, -n N
                            noise level for thresholding (default: 300)
      --nthreads T, -t T    number of threads to be used for fitting (default: 12)
      --plot-level L, -p L  do a live plot of the fitting (L=0: no plot, L=1: show the images, L=2: show images and signals)
      --t2-limits min max   set the limits for t2 calculation (default: 20-80)
      --b1-limits min max   set the limits for b1 calculation (default: 0.5-1.2)
      --use-gpu, -g         use GPU for fitting
      --ff-map dir, -m dir  load a fat fraction map
      --register-ff, -r     register the fat fraction dataset
      --etl-limit N, -e N   reduce the echo train length
      --out-suffix ext, -s ext
                            add a suffix to the output map directories
      --slice-range start end, -l start end
                            Restrict the fitting to a subset of slices
      --refocusing-width factor, -w factor
                            Slice width of the refocusing pulse with respect to the excitation (default 1.2) (Siemens standard)
      --exc-profile path    Path to the excitation slice profile file
      --ref-profile path    Path to the refocusing slice profile file

### Dataset

The dataset must be a directory containing 2D DICOM images from a multiecho spin echo acquisition,
ordered as (slice1echo1 - slice2echo1 - ... - slice1echo2 - slice2echo2 - ...)

### Slice profile

An accurate slice profile is crucial to obtain accurate results. By default,
a hanning-windowed sinc pulse is used, and the refocusing pulse has a 1.2x the
slice width of the excitation pulses. This reflects the parameters of the
standard Siemens Spin Echo sequence.

External slice profile files can be provided. They are text files with angle
values (in degrees) across half the profile (i.e. starting with ~90 or ~180 and
decreasing), with one value per line.

Either none or both slice profiles must be given, and both slice arrays must
contain the same number of samples.

Example:

    90
    89.8014
    89.2106
    88.2423
    86.9198
    85.2734
    83.3385
    81.1537
    78.7585
    76.1921
    ...

# Acknowledgments
EPG code was made possible by Matthias Weigel.

This project was supported by the [SNF](http://www.snf.ch/) (grant number 320030_172876)


> Written with [StackEdit](https://stackedit.io/).
