LSDSAR-line segment detector for SAR images

version 1.1-June 27, 2018
by Chenguang Liu 
Email: chenguangl@whu.edu.cn

Description: 

LSDSAR is a line segment detector for SAR images described in the paper

"LSDSAR, a Markovian a contrario framework for line segment detection in SAR images",
 by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. (Pattern Recognition, 2019)

Files

README.txt                      -This file
COPYING                          -GNU AFFERO GENERAL PUBLIC LICENSE Version 3.
lsdsar.c                             -LSDSAR module ANSI C code.
lsdsar.h                             -LSDSAR module ANSI C header.
mexlsdsar.c                      -mexFunction for the use of LSDSAR in matlab.
demo_purenoise.m            - a matlab demo file.
demo_real_image.m          - a matlab demo file.
demo_synthetic_image.m  - a matlab demo file.
image                               - a folder including SAR images
precompute_transition.m   - precomputed transition probabilites for the first order Markov chain.
sarimshow.m                    - a matlab function to show SAR images.

To compile it, please type "mex mexlsdsar.c lsdsar.c", then you can run any demo file to test it.

There are three demo files:
    demo_purenoise: the number of false detections in pure noise image by LSDSAR is tested.
    demo_synthetic_image: line segment detection results in synthetic images by LSDSAR are given.
    demo_real_image: line segment detection results in real SAR images by LSDSAR are given.

I would be grateful to receive any advices or possible errors in the source code.

Chenguang Liu
Telecom Paris
Email: chenguang.liu@telecom-paris.fr
Email: chenguangl@whu.edu.cn (permanent) 

Copyright and License

LSDSAR-line segment detector for SAR images.

This code is with the publication below:

 "LSDSAR, a Markovian a contrario framework for line segment detection in SAR images",
 by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. Pattern Recognition, 2019.

*NOTICE: This program is modified from the source code of LSD:
*"LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
*Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
*Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
*http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
*Date of Modification: 27/06/2018.

*NOTICE: This program is released under GNU Affero General Public License
*and any conditions added under section 7 in the link:
*https://www.gnu.org/licenses/agpl-3.0.en.html

Copyright (c) 2017, 2018 Chenguang Liu <chenguangl@whu.edu.cn>

LSDSAR is free software: you can redistribute it and/or modify 
 it under the terms of the GNU General Public License as published 
 by the Free Software Foundation, either version 3 of the License, 
 or (at your option) any later version.

LSDSAR is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.
 
You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

Thanks.


===============================================
===============================================
==NOTICE: HERE IS THE README FILE OF THE ORIGINAL LSD
===============================================
==LSD - Line Segment Detector
===========================
==
==Version 1.6 - November 11, 2011
==by Rafael Grompone von Gioi <grompone@gmail.com>
==
==
==Introduction
==------------
==
==LSD is an implementation of the Line Segment Detector on digital
==images described in the paper:
==
==  "LSD: A Fast Line Segment Detector with a False Detection Control"
==  by Rafael Grompone von Gioi, Jeremie Jakubowicz, Jean-Michel Morel,
==  and Gregory Randall, IEEE Transactions on Pattern Analysis and
==  Machine Intelligence, vol. 32, no. 4, pp. 722-732, April, 2010.
==
==and in more details in the CMLA Technical Report:
==
==  "LSD: A Line Segment Detector, Technical Report",
==  by Rafael Grompone von Gioi, Jeremie Jakubowicz, Jean-Michel Morel,
==  Gregory Randall, CMLA, ENS Cachan, 2010.
==
==The version implemented here includes some further improvements as
==described in the IPOL article of which this file is part:
==
==  "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
==  Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
==  Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
==  http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
==
==Copyright and License
==---------------------
==
==Copyright (c) 2007-2011 rafael grompone von gioi <grompone@gmail.com>
==
==LSD is free software: you can redistribute it and/or modify
==it under the terms of the GNU Affero General Public License as
==published by the Free Software Foundation, either version 3 of the
==License, or (at your option) any later version.
==
==LSD is distributed in the hope that it will be useful,
==but WITHOUT ANY WARRANTY; without even the implied warranty of
==MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
==GNU Affero General Public License for more details.
==
==You should have received a copy of the GNU Affero General Public License
==along with this program. If not, see <http://www.gnu.org/licenses/>.
================================================
NOTICE: LSDSAR README FILE IS ON THE TOP OF THIS FILE
================================================
