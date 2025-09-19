/* Macros for the header version.
 */

#ifndef VIPS_VERSION_H
#define VIPS_VERSION_H

#define VIPS_VERSION "8.17.2"
#define VIPS_VERSION_STRING "8.17.2"
#define VIPS_MAJOR_VERSION (8)
#define VIPS_MINOR_VERSION (17)
#define VIPS_MICRO_VERSION (2)

/* The ABI version, as used for library versioning.
 */
#define VIPS_LIBRARY_CURRENT (61)
#define VIPS_LIBRARY_REVISION (2)
#define VIPS_LIBRARY_AGE (19)

#define VIPS_CONFIG "enable debug: false\nenable deprecated: false\nenable modules: false\nenable C++ binding: true\nenable RAD load/save: false\nenable Analyze7 load: false\nenable PPM load/save: false\nenable GIF load: true\nFFTs with fftw: false\nSIMD support with libhwy: true\nICC profile support with lcms2: true\ndeflate compression with zlib: true\ntext rendering with pangocairo: true\nfont file support with fontconfig: true\nEXIF metadata support with libexif: true\nJPEG load/save with libjpeg: true\nJXL load/save with libjxl: false (dynamic module: false)\nJPEG2000 load/save with OpenJPEG: false\nPNG load/save with spng: true\nimage quantisation with imagequant: true\nTIFF load/save with libtiff-4: true\nimage pyramid save with libarchive: true\nHEIC/AVIF load/save with libheif: true (dynamic module: false)\nWebP load/save with libwebp: true\nPDF load with PDFium or Poppler: false (dynamic module: false)\nSVG load with librsvg-2.0: true\nEXR load with OpenEXR: false\nWSI load with OpenSlide: false (dynamic module: false)\nMatlab load with Matio: false\nNIfTI load/save with libnifti: false\nFITS load/save with cfitsio: false\nGIF save with cgif: true\nMagick load/save with MagickCore: false (dynamic module: false)"

/* Not really anything to do with versions, but this is a handy place to put
 * it.
 */
#define VIPS_ENABLE_DEPRECATED 0

#endif /*VIPS_VERSION_H*/
