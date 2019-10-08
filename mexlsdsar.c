/*----------------------------------------------------------------------------
LSDSAR-line segment detector for SAR images.

This code is with the publication below:

 "LSDSAR, a Markovian a contrario framework for line segment detection in SAR images",
 by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. (Pattern Recognition, 2019)

*NOTICE: This code is modified from the source code of LSD:
*"LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
*Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
*Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
*http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
*Date of Modification: 27/06/2018.

*NOTICE: This program is released under GNU Affero General Public License
*and any conditions added under section 7 in the link:
*https://www.gnu.org/licenses/agpl-3.0.en.html

Copyright (c) 2017, 2018 Chenguang Liu <chenguangl@whu.edu.cn>

This program is free software: you can redistribute it and/or modify 
 it under the terms of the GNU General Public License as published 
 by the Free Software Foundation, either version 3 of the License, 
 or (at your option) any later version.

This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.
 
You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------------------

*NOTICE: This code is modified from the source code of LSD:
*"LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
*Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
*Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
*http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
*Date of Modification: 27/06/2018.

The modifications lie in functions:
    1) double * lsd(int * n_out, double * img, int X, int Y),
    2) double * lsd_scale(int * n_out, double * img, int X, int Y, double scale),
    3) double * lsd_scale_region( int * n_out,
                           double * img, int X, int Y, double scale,
                           int ** reg_img, int * reg_x, int * reg_y ),
    4)double * LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double scale, double sigma_scale, double quant,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y ),
    5) static image_double ll_angle( image_double in, double threshold,
                              struct coorlist ** list_p, void ** mem_p,
                              image_double * modgrad, unsigned int n_bins ),
    6) static int refine( struct point * reg, int * reg_size, image_double modgrad,
                   double reg_angle, double prec, double p, struct rect * rec,
                   image_char used, image_double angles, double density_th ),
    7) static int reduce_region_radius( struct point * reg, int * reg_size,
                                 image_double modgrad, double reg_angle,
                                 double prec, double p, struct rect * rec,
                                 image_char used, image_double angles,
                                 double density_th ),
    8) static double rect_improve( struct rect * rec, image_double angles,
                            double logNT, double log_eps ),
    9) static double rect_nfa(struct rect * rec, image_double angles, double logNT),
    10) static double nfa(int n, int k, double p, double logNT).

The other functions of the code are kept unchanged.

 I would be grateful to receive any advices or possible erros in the source code. 

Chenguang Liu
Telecom ParisTech
Email: chenguang.liu@telecom-paristech.fr
Email: chenguangl@whu.edu.cn (permanent)
*/

#include <stdio.h>
#include <stdlib.h>
#include "lsdsar.h"
#include"mex.h"

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  double * image;
  double * inputv;
  double * out;
  double *outdata;
  int x,y,i,j,n;
  int X;  /* x image size */
  int Y;  /* y image size */
int Xin,Yin;
  /* create a simple image: left half black, right half gray */
  image=mxGetPr(prhs[0]);
  inputv=mxGetPr(prhs[1]);
  Xin=mxGetM(prhs[1]);
  Yin=mxGetN(prhs[1]);
  X=mxGetM(prhs[0]);
  Y=mxGetN(prhs[0]);
  
  if( image == NULL )
    {
      fprintf(stderr,"error: not enough memory\n");
      exit(EXIT_FAILURE);
    }


  /* LSD call */
  
  out = lsd(&n,image,X,Y,inputv);
  int mm=7;
  plhs[0]=mxCreateDoubleMatrix(n,mm,mxREAL);
  outdata=mxGetPr(plhs[0]);
  


  /* print output */
  printf("%d line segments found:\n",n);
  for(i=0;i<n;i++)
    {
      for(j=0;j<mm;j++)
	outdata[i+j*n]=out[i*mm+j];
     
    }

  /* free memory */
  free((void *)  out);
}
