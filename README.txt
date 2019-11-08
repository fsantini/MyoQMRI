# MyoQMRI
This project is an open-source effort to put together tools for quantitative MRI of the muscles.

## Multi-Echo Spin Echo

The main function is used to obtain **water T2** and **fat fraction** from multi-echo spin echo images as described in [Marty et al. ](https://doi.org/10.1002/nbm.3459)

This implementation uses the GPU for the generation of a dictionary of signals through [Extended Phase Graph simulation](https://doi.org/10.1002/jmri.24619).

### Usage

    usage: epgFit.py [-h] [--fat-t2 T2] [--noise-level N] [--nthreads T]
                     [--plot-level L] [--t2-limits min max] [--b1-limits min max]
                     [--use-gpu TF]
                     path
    
    Fit a multiecho dataset
    
    positional arguments:
      path                  path to the dataset
    
    optional arguments:
      -h, --help            show this help message and exit
      --fat-t2 T2, -f T2    fat T2 (default: 151)
      --noise-level N, -n N
                            noise level for thresholding (default: 300)
      --nthreads T, -t T    number of threads to be used for fitting (default: 4)
      --plot-level L, -p L  do a live plot of the fitting (L=0: no plot, L=1: show
                            the images, L=2: show images and signals)
      --t2-limits min max   set the limits for t2 calculation (default: 20-80)
      --b1-limits min max   set the limits for b1 calculation (default: 0.4-1.4)
      --use-gpu TF, -g TF   use GPU for fitting (0 == false)

# Acknowledgments
EPG code was made possible by Matthias Weigel
This project was supported by the [SNF](http://www.snf.ch/) (grant number 320030_172876)


> Written with [StackEdit](https://stackedit.io/).
