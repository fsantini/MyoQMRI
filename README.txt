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
