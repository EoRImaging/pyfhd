# IDL to Python Translation Guide

This part of the guide is going to be more personal and contains lessons from my
own (Joel Dunstan, [SkyWa7ch3r](https://github.com/SkyWa7ch3r)) approx. 3 years
experience of translating the majority of `FHD` from IDL to Python to create `pyfhd`.

## Introduction

Hello there weary traveler, you wish to translate from IDL to Python or have been
tasked with doing so, first thing my sincerest apologies for your future sanity
and also to your colleagues as you likely vent to them about the annoyances of
doing the translation. Here I'll try to put all the discoveries, roadbumps and
potholes that will slow you down along the way. Be prepared to laugh, sigh, be
filled with rage and maybe shed a tear if you find out a problem you've been
facing for a week is because of some <del>dumb</del> *interesting* behaviour.

## Getting IDL

First thing, you'll need access to IDL, this is easier said than done as you
require a license to use the language if you're lucky your instituion has an IDL
license or the HPC you use has an IDL license. If you plan on installing IDL to
your local machine you will need to contact
[NV5 Geospatial](https://www.nv5geospatialsoftware.com/Products/IDL) directly to
get a license and get the download link through their locked download links. If
you're institution has a license server you should be able to get a license from
the server and an install package from your institution. There are open source
versions of IDL, [`GDL`](https://github.com/gnudatalanguage/gdl) and
[`Fawlty`](https://www.flxpert.hu/fl/), however in my experience they didn't
support running all the functions needed for `FHD` or produced different results
to what's expected even in simple scenarios, hopefully that has changed and you
can try to use them, feel free to let me know how that went.

Now that you have IDL, you'll need to follow the
[FHD install steps](https://github.com/EoRImaging/FHD/tree/master#installation).
You'll notice that all the functions get added to the IDL path, IDL only really
has one namespace for functions, do keep that in mind unlike Python which has
packages/libraries, modules/subpackages which divide up the name spaces and
require explicit imports. 

## FHD Dependency Equivalents

Now in terms of matching up some of the `FHD` dependencies and `pyfhd` dependencies,
anything from the `FHD`, `fhdps_utils`, and `pipeline_scripts` repositories will
require straight translation line by line. For the `IDLAstro` library from NASA,
a mix of `astropy`, `numpy` and `scipy` should allow you to either drectly replace
functions when they appear in the `FHD` code or allow much easier translation/rewriting
of the function. For the `IDLAstro` library all the `pro` files are publicly
available so if there is a need to directly translate the function into Python
that is possible. `coyote` does much of the graphical heavy lifting in `FHD`, for
any plots inside of `pyfhd`, the plotting/imaging libraries `matplotlib`,
`seaborn`, `plotly`, `Pillow` and `OpenCV` should get you what you need to make
almost any visualization you wish to make (`plotly` is excellent when you need
interactivity). For `HEALPix` you have the `healpy` library maintained by the
same organization, so functions should have similar names and similar purposes
without any translation needed.

## The Right Tools for the Right Job

In terms of tools you can use to translate IDL to Python, after all this time,
unfortunately for you, I've decided the best tool is **you**. None of the tools
out there like [`idl2py`](https://github.com/dnidever/idl2py) successfully
translate to Python in a way that makes it faster than you translating the
function line by line, they even occasionally miss things like `FLTARR` which
translate perfectly into `np.zeros`. To be fair on these tools its an almost
impossible task to automate because in IDL keywords in functions can
*entirely change how the function works* thus any code you make to automatically
translate will be full of exceptions and will still have problems translating
pointer arrays due to the shapes of said arrays not always being easy to know
ahead of time. Unfortunately, not even ChatGPT or LLMs can completely help you
either, although to be honest they do a somewhat decent job, feel free to
experiment with them and let me know how it goes, although I suspect it will miss
some of the interesting behaviour that I will point out later. If you're using
VSCode, NV5 recently brought out an official VSCode extension for IDL, check it
out [here](https://marketplace.visualstudio.com/items?itemName=IDL.idl-for-vscode).
IDL for VSCode will give you proper syntax highlighting for `pro` files, and
hopefully they will continue to increase it's usefulness in the future.

Another pair of tools that are worth talking about is the `IDL to Python Bridge`
and `Python to IDL Bridge` that is provided as a part of the IDL package. The
`Python to IDL Bridge` can provide some nice to have features such as running
IDL code in a Jupyter notebook which can be particularly useful for debugging
some IDL to Python translations because you can rerun just parts of the code
you're having issues with translating. The `IDL to Python Bridge` does also make
it possible to use Python within IDL, while the `Python to IDL Bridge` makes it
possible to use IDL in Python. At the time of writing this its now possible to
pass objects to and from IDL in the `Python to IDL Bridge`, however I cannot
confirm if you can still only run the built-in IDL functions in the
`Python to IDL Bridge`. When I previously used the bridge I wasn't able to run
functions made inside of `FHD` such as `weight_invert` (I suspected this was due
to the IDL_PATH being setup everytime you re-ran IDL.run for every command).
Hopefully if you try the bridges on the latest version of IDL, you may have
better luck meaning you could have more power for debugging than I did because
you can directly run the IDL code side by side and directly compare the IDL and
Python outputs without the need for `sav` files. Feel free to let me know how
this changes or to update this section yourself.

## The Hidden Gems in IDL to make you tear your hair out

There are several gotchas hidden thoughout IDL, usually most of them won't appear
to an unexperienced IDL person who hasn't been burned by them before until they
try to run the Python equivalent, for example, me and hopefully not you. I will
try to list as many as I can here, lot of these can be found in IDL documentation
as extra notes or footnotes (rather than a big notice saying "Hey this is important,
if you use this function this way, this function is entirely different" which
would be a little nicer).

### Column Based vs Row-Based Indexing
IDL is column based while Python is row based, meaning you will likely need to
change the shape of certain arrays for the multiplication of arrays and matrices
to work as you expect. It's the reason why the visibility arrays go by polarization
then frequency then baselines in pyfhd compared to polarization then baselines
then frequencies in `FHD` as it allows us to translate the indexing as it's done
in `FHD` without having to constantly swap indexes around during translation
which gets confusing.

### Single Precision wins everytime
IDL is single precision even if it kills itself, in many cases even if you
specify double precision using the `/DOUBLE` keyword, there's bound to be some
function somewhere that is restricted to single precision. This is slowly changing.

### Input or Output...why not both? (Function Parameters)
Function parameters in IDL can be input **or** output! If during your translation
a variable exists seemingly without being initialized somewhere, check the function
calls, parameters store the results, this is the case with many IDL functions,like
`HISTOGRAM` where the `REVERSE_INDICES=ri` part of the call actually *returns*
the reverse indices into an `ri` variable. This is also used extensively for
many `FHD` functions and can get confusing, in Python doing returns this way is
possible but for the sake of being consistent, any and all returns are done with
an explicit `return` statement and put into the docstrings as a return.  

### Python Subsetting vs IDL subsetting
When dealing with how IDL deals with subsetting, it's important to remember than
IDL is inclusive of *both* the start and the end index, i.e. if we index an array
between the 0th value and the 3rd value, we will get the 0th value and all the
values upto and including the 3rd value. In Python the subsetting is only inclusive
of the start but not the end index, i.e. if we index an array between the 0th
value and the 3rd value, we will get the 0th value and all the values upto the
3rd value, but not including the third value.
    
    IDL
    ```idl
    IDL> test = [1,2,3,4,5]
    IDL> test[0:3]
           1       2       3       4
    ```

    Python
    ```python
    >>> test = np.array([1,2,3,4,5])
    >>> test[0:3]
    array([1, 2, 3])
    ```

    The inclusion of both ends in IDL also works the same for loops of any kind,
    so when translating you should be careful with how you index the array at the
    end primarily or the end conditions for your loop. You will find many users
    of IDL have already taken this into account and loop `n` times going from
    `0` to `n - 1`, so in python using the `range(n)` function should do the trick.

### What the /EVEN (IDL Median)
The median function in IDL doesn't do the average between the two middle numbers
by default when dealing with an array of even size unless you use the `/EVEN`
keyword, it will take the maximum of the two numbers when dealing with an even
sized array be default. This has often made a difference when dealing with close
to single precision values, in my experience you're usually better off using
`np.median` in most cases and accepting the differences, there has only been one
notable case where we needed the IDL median over `np.median` and that's during
the calibration process (as the comparison to `conv_thresh` would fail due to the
precision differences).

### MATRIX_MULTIPLY: Is it a dot product or outer product, you'll find out at run time!
Matrix Multiply does the dot product for matrices, and because IDL is column
based while Python is row based, you will need to switch around the order of
multiplication in Python compared to IDL. Furthermore, matrix multiply does the
outer product when you make on the provided matrices/arrays 1D array. So everytime
you see matrix multiply you'll likely have to check at runtime during testing
if the multiply is meant to be the dot product or outer product.

### NaN Shenanigans (IDL total and mean)
In the case of an array containing only `NaNs` there is a slight difference
between IDL's `total` and `mean`, where  `total(x,/nan)` will give you `0` while
`mean(x, /nan)` will give you a `NaN`

### COMPLEX ATAN
In IDL `ATAN` when used with complex numbers and the `/PHASE` keyword gives you
the phase of the imaginary number. In Python the equivalent function is `np.angle`.
Do not be tempted to compute it using `np.atan` -- it can give you NaNs unexpectedly.

### DIVIDE BY ZERO: 1/0 + 10 = 11 Folks
When doing a divide by 0 in IDL, it does produce an error
`% Program caused arithmetic error: Integer divide by 0` however it doesn't
actually break the program, IDL continues to work, watch out for it! Don't ask
me why they've done it...and if you don't believe me here is an example in IDL 8.8.0:
  ```idl
  IDL> test = [[1.,2.],[3.,4.]]
  IDL> test[0] = 1/0 + 10
  % Program caused arithmetic error: Integer divide by 0
  IDL> test
  11.000000       2.0000000
  3.0000000       4.0000000
  ```

### Indexing beyond the size of the array will work
Indexing arrays using other arrays can also lead to <del>dumb</del> *interesting*
behaviour, if we follow our test array again of 4 values and we try to access the
5th element of the array we expectedly get an error.
    ```idl
    IDL> test = [[1.,2.],[3.,4.]]
    IDL> test[5]
    % Attempt to subscript TEST with <INT      (       5)> is out of range.
    % Execution halted at: $MAIN$ 
    ```
    However if we put that index into another array, let's call it `idx` and then
    access the `test` array with the `idx` array we get the following:
    ```idl
    IDL> test = [[1.,2.],[3.,4.]]
    IDL> idx = [5]
    IDL> test[idx]
       4.0000000
    ```
    Yep that's the last value of the array, and to really get the ball into the endzone:
    ```idl
    IDL> idx = indgen(10) + 1000
    IDL> idx
      1000    1001    1002    1003    1004    1005    1006    1007    1008    1009
    IDL> test[idx]
       4.0000000       4.0000000       4.0000000       4.0000000       4.0000000       4.0000000       4.0000000
       4.0000000       4.0000000       4.0000000
    ```
    In summary, watch out for this behaviour, it usually comes up in C when you
    use pointers beyond their designated memory addresses, resulting in overflows,
    however for whatever reason IDL does their own version of overflow.
    In Python, this functionality of indexing beyond the array, does not work,
    and is expected to never work (and it never should).

### Poly fit/val
The `POLY_FIT` function in IDL has the ability to do both `polyfit` and `polyval`,
so watch out for the use of the `YFIT` keyword. In general, `poly_fit(a, b, 2)`
in IDL is `np.polynomial.polynomial.Polynomial.fit(a, b, deg=2).convert().coef`
in Python. If you see the `YFIT` keyword, then it's a case of calling
`np.polynomial.polynomial.polyval` on the calculated coefficients i.e.
`phase_fit = np.polynomial.polynomial.polyval(a, np.polynomial.polynomial.Polynomial.fit(a, b, deg=2).convert().coef)`.

### Standard Linear Algebra Shenanigans (LA_LEAST_SQUARES)
`LA_LEAST_SQUARES` in IDL and `np.linalg.lstsq` in Python (and the SciPy least
square functions as well) will produce different results due to the solvers in
use, and the assumption that the design matrix given to the solvers is full rank
in `LA_LEAST_SQUARES` while NumPy and SciPy assuming that the design matrix is
not full rank. Even using the same solvers doesn't give the same results, so it
might come down to how each of the functions were compiled, either way, keep in
consideration they just won't get the same results in most cirumstances.

### REBIN
The `REBIN` has been replicated in `pyfhd`, however it's use should be used
sparingly as its doing interpolation when increasing an array in size, and
averaging when decreasing the size of an array i.e. it's upscaling or downscaling
and it's treating your array or matrix as an image rather than any ordinary array.
In many cases it could be best to use `np.tile`, `np.pad` or `np.repeat` to do
the same task more consistently.

### That's a smooth move (IDL SMOOTH)
The `SMOOTH` function in IDL is a boxcar averaging function so it should be
replaced with `scipy.ndimage.uniform_filter` and the `/edge` keywords will
dictate the mode you need to use for the `uniform_filter` function.

### Where is COMPLEX? (IDL WHERE and COMPLEX numbers)
When dealing with complex numbers in IDL, it's possible to get different results
when using `WHERE` in IDL and `np.where` as `WHERE` in IDL applies to the
absolute values of the numbers, while `np.where` looks at just the real numbers
for example:
  
  IDL
  ```idl
  IDL> test = COMPLEX([1,2,0],[1,2,3])
  IDL> WHERE(test gt 1)
           0           1           2
  ```

  Python
  ```python
  >>> test = np.array([1 + 1j, 2 + 2j, 0 + 3j])
  >>> np.where(test > 1)
  (array([0, 1]),)
  >>> np.where(np.abs(test) > 1)
  (array([0, 1, 2]),)
  ```

  It has only affected my translation once, but something to be aware of.

### The Point (IDL Pointers)
Pointers are used in IDL a lot, especially in `FHD`, when translating pointer
arrays my advice is to try and find out the full shape of the array and set the
dtype appropriately, rather than creating an `object` array or `list` to represent
the pointer array. `object` arrays in NumPy usually lead to more problems than
they're worth, and make using the built in vectorized functions a real pain.

### I feel sorry for the person that had to make this function (scipy.io.readsav)
There is a `readsav` function in `scipy.io` that gives you the ability to read
in `sav` files, which is great that it exists. Unfortunately though, `readsav`
is usually quite slow due to IDL's differences to other languages down to the
byte level, as such it's having to loop through all the bytes to get the necessary
bits out (poor thing, imagine having to read a 44GB file
*several bytes at a time in a python loop*). `readsav` does read the `sav` file
into a python dictionary which sounds good, except that it turns structures that
were saved into the `sav` files into `np.recarrays` or NumPy record arrays. NumPy
record arrays are often slower than dictionaries to access and in the case of
`pyfhd` offer no benefits over dictionaries (in fact I'd argue a dictionary is
better is almost all circumstances). `readsav` also usually makes a mess of
pointer arrays too by turning pointer arrays into `object` arrays which each
index contains a `numpy.ndarray`, meaning to access an array you might have to
do something like `sav_file['array'][0]array[0][0]` to access the first numpy
array in the array of arrays. As such I developed the `pyfhd.io.recarray_to_dict`
function which can take in a `np.recarray` or `dict` and turn any record arrays
into a `dict`. `recarray_to_dict` works recursively too, so any sub-record arrays
will also turn into a `dict`, furthermore, additional formatting of arrays will
take place to turn any `object` arrays into a proper NumPy array with a numpy
data type (dtype). `recarray_to_dict` also deals with the values that are scalar
values when loaded in from the `sav` file as well, `readsav` usually turns scalar
values into arrays with a single vaue which are inconvenient to access.
`recarray_to_dict` is always a work in progress, happy to try and edit the
function as necessary.
* If you're dealing with large pointer arrays that can't convert to a proper
dtype array due their size, e.g. the `beam_ptr` complex array was an example of
this with a size of `2*384*8128*51*51*2916*16 bytes` or `~757.5 TB`, then ideally
you need to convert these into HDF5 (with `h5py`) files which can allow you to
chunk large datasets. It's also worth checking that there aren't pointers
referencing other pointers, this was the case with `beam_ptr` as such the actual
size ended up being `2*384*51*51*2916*16 bytes` or `~93.2 GB` instead. It's also
possible to use other frameworks like `Dask` to achieve what we're doing with
`h5py`, but we can use `h5py` without having to worry about compatiblity with
NumPy functions.

### Concatenating Arrays
In IDL concatenating arrays can be done in many ways, one of the ways you may see is like so:
  ```idl
  IDL> test = [1,2,3]
  IDL> test = [test, test]
  IDL> test
         1       2       3       1       2       3
  ```
  So be on the look out for that sort of behaviour.

### It's the minimum and maximum I can do (IDL < and > operators)
The `>` and `<` operators in IDL do the same behaviour as `np.maximum` and
`np.minimum` respectively, do not get them confused with the `GT` (greater than)
and `LT` (less than) operators in IDL. Furthermore, the `>` and `<` operators in
IDL can be chained together, I'll provide many examples below:
    ```idl
    IDL> test = [-1,2,3,-4,5]
    IDL> test > 0
       0       2       3       0       5
    IDL> test < 0
      -1       0       0      -4       0
    IDL> test < 0 > 3
       3       3       3       3       3
    IDL> test < 1 > 0
       0       1       1       0       1
    ```

    The bottom example is the most common example you'll see in `FHD` as it
    turns all numbers above 1 into 1, and the numbers below 0 into 0 allowing to
    quickly make a flagged array. To show the same examples in Python look below:

    ```python
    >>> test = np.array([-1,2,3,-4,5])
    >>> np.maximum(test, 0)
    array([0, 2, 3, 0, 5])
    >>> np.minimum(test, 0)
    array([-1,  0,  0, -4,  0])
    >>> np.maximum(np.minimum(test, 0), 3)
    array([3, 3, 3, 3, 3])
    >>> np.maximum(np.minimum(test, 1), 0)
    array([0, 1, 1, 0, 1])
    ```

## Thanks for Reading, and Ye Be Warned

I hope this contribution guide helps you in your translation efforts with much
less pain than me, and all of this helps you get your function into Python as
quick as possible. Translation and testing of IDL to Python code can be a
frustrating task even at the best of times. So...

![Translators Ye Be Warned](ye_be_warned_smaller.gif)