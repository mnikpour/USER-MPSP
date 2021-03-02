/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
This version of the code is updated to incorporate 
the multi-particle, multi-phase battery slurry specifications.
 ------------------------------------------------------------------------- */
#include <math.h>
#include <stdlib.h>
#include "pair_sph_taitwater_morris_comb.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"


#include "update.h"
#include "neigh_request.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSPHTaitwaterMorrisComb::PairSPHTaitwaterMorrisComb(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  first = 1;
}

/* ---------------------------------------------------------------------- */

PairSPHTaitwaterMorrisComb::~PairSPHTaitwaterMorrisComb() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(rho0);
    memory->destroy(soundspeed);
    memory->destroy(B);
    memory->destroy(viscosity);
  }
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairSPHTaitwaterMorrisComb::init_style() {
  // need a full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ---------------------------------------------------------------------- */

void PairSPHTaitwaterMorrisComb::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih, ihsq, velx, vely, velz;
  double rsq, tmp, wfd, delVdotDelR, deltaE;
  double wf, r, divergence, changeInDensity, rhoi[60000];

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **v = atom->vest;
  double **x = atom->x;
  double **f = atom->f;
  double *rho = atom->rho;
  double *mass = atom->mass;
  double *de = atom->de;
  double *drho = atom->drho;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  // BYU modeling: defining static variable to access the values form previous timestep (last calculated values) for 2000000 particles
  static double previousVelocity[60000][3];
  static double previousPosition[60000][3];
  static bool initialPositionsUpdated = false;
  static double density[60000], Densitychange[60000] ;
  // check consistency of pair coefficients

  if (first) {
    for (i = 1; i <= atom->ntypes; i++) {
      for (j = 1; i <= atom->ntypes; i++) {
        if (cutsq[i][j] > 1.e-32) {
          if (!setflag[i][i] || !setflag[j][j]) {
            if (comm->me == 0) {
              printf(
                  "SPH particle types %d and %d interact with cutoff=%g, but not all of their single particle properties are set.\n",
                  i, j, sqrt(cutsq[i][j]));
            }
          }
        }
      }
      if (!initialPositionsUpdated){
         previousPosition[i][0] = x[i][0];
         previousPosition[i][1] = x[i][1];
         previousPosition[i][2] = x[i][2];
        
         previousVelocity[i][0] = v[i][0];
         previousVelocity[i][1] = v[i][1];
         previousVelocity[i][2] = v[i][2];
        
         density[i] = rho[i];
         //printf("initial position: %e, %e, %e\n\rvelocity: %e, %e, %e\n\r densities updated\n\r");
     }
  }
  initialPositionsUpdated = true;
  first = 0;
 }


  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
//printf( " inum = %i \n", inum);
//** initialize quantities with self-contribution,
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    imass = mass[itype];

    h = cut[itype][itype];
    if (domain->dimension == 3) {
      
      // Lucy kernel, 3d
      wf = 2.0889086280811262819e0 / (h * h * h);
      

      // quadric kernel, 3d
      //wf = 2.1541870227086614782 / (h * h * h);
    } else {
      
      // Lucy kernel, 2d
      wf = 1.5915494309189533576e0 / (h * h);
      

      // quadric kernel, 2d
      //wf = 1.5915494309189533576e0 / (h * h);
    }

    rho[i] = imass * wf;
  	

  }

  // add density at each atom via kernel function overlap
  for (ii = 0; ii < inum; ii++) {

    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      jtype = type[j];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutsq[itype][jtype]) {
        h = cut[itype][jtype];
        ih = 1.0 / h;
        ihsq = ih * ih;

        if (domain->dimension == 3) {
          
          // Lucy kernel, 3d
          r = sqrt(rsq);
          wf = (h - r) * ihsq;
          wf =  2.0889086280811262819e0 * (h + 3. * r) * wf * wf * wf * ih;
          

          // quadric kernel, 3d
          //wf = 1.0 - rsq * ihsq;
          //wf = wf * wf;
          //wf = wf * wf;
          //wf = 2.1541870227086614782e0 * wf * ihsq * ih;
        } else {
          // Lucy kernel, 2d
          r = sqrt(rsq);
          wf = (h - r) * ihsq;
          wf = 1.5915494309189533576e0 * (h + 3. * r) * wf * wf * wf;

          // quadric kernel, 2d
          //wf = 1.0 - rsq * ihsq;
          //wf = wf * wf;
          //wf = wf * wf;
          //wf = 1.5915494309189533576e0 * wf * ihsq;
        }

        rho[i] += mass[jtype] * wf;

      }

    }
    //printf( " rho[i] = %f \n ", rho[i]);


  }
//printf( " inum = %f \n ", inum);
  // communicate densities
  //comm->forward_comm_pair(this);

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
  	
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    imass = mass[itype];
    divergence = (v[i][0] - previousVelocity[i][0])/(x[i][0] - previousPosition[i][0]) +
  (v[i][1] - previousVelocity[i][1])/(x[i][1] - previousPosition[i][1]) +
  (v[i][2] - previousVelocity[i][2])/(x[i][2] - previousPosition[i][2]);
      Densitychange[i] = rho[i] - density[i];

    // compute pressure of atom i with Tait EOS
    tmp = rho[i] / rho0[itype];
    fi = tmp * tmp * tmp;
    fi = B[itype] * (fi * fi * tmp - 1.0) / (rho[i] * rho[i]);

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = mass[jtype];

      if (rsq < cutsq[itype][jtype]) {
        h = cut[itype][jtype];
        ih = 1.0 / h;
        ihsq = ih * ih;

        wfd = h - sqrt(rsq);
        if (domain->dimension == 3) {
          // Lucy Kernel, 3d
          // Note that wfd, the derivative of the weight function with respect to r,
          // is lacking a factor of r.
          // The missing factor of r is recovered by
          // (1) using delV . delX instead of delV . (delX/r) and
          // (2) using f[i][0] += delx * fpair instead of f[i][0] += (delx/r) * fpair
          wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih;
        } else {
          // Lucy Kernel, 2d
          wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
        }

        // compute pressure  of atom j with Tait EOS
        tmp = rho[j] / rho0[jtype];
        fj = tmp * tmp * tmp;
        fj = B[jtype] * (fj * fj * tmp - 1.0) / (rho[j] * rho[j]);

        velx=vxtmp - v[j][0];
        vely=vytmp - v[j][1];
        velz=vztmp - v[j][2];

        // dot product of velocity delta and distance vector
        delVdotDelR = delx * velx + dely * vely + delz * velz;

        fvisc = viscosity[itype][jtype] / (rho[i] * rho[j]);  

        fvisc *= imass * jmass * wfd;
	
        // total pair force & thermal energy increment
        fpair = -imass * jmass * (fi + fj) * wfd;
        deltaE = -0.5 *(fpair * delVdotDelR + fvisc * (velx*velx + vely*vely + velz*velz));


        f[i][0] += delx * fpair + velx * fvisc;
        f[i][1] += dely * fpair + vely * fvisc;
        f[i][2] += delz * fpair + velz * fvisc;

        // and change in density
        drho[i] += jmass * delVdotDelR * wfd;

        // change in thermal energy
        de[i] += deltaE;

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair + velx * fvisc;
          f[j][1] -= dely * fpair + vely * fvisc;
          f[j][2] -= delz * fpair + velz * fvisc;
          de[j] += deltaE;
          drho[j] += imass * delVdotDelR * wfd;
        }


        if (evflag)
          ev_tally(i, j, nlocal, newton_pair, 0.0, 0.0, fpair, delx, dely, delz);
                  
    


  
    }

    }
    
    //BYU modeling: assigning variables for divergence and density change calculations:
   
    //Densitychange[i] = changeInDensity;  
 //printf( " rho[i] = %f \n ", rho[i]);

    previousVelocity[i][0] = v[i][0];
    previousVelocity[i][1] = v[i][1];
    previousVelocity[i][2] = v[i][2];
    previousPosition[i][0] = x[i][0];
    previousPosition[i][1] = x[i][1];
    previousPosition[i][2] = x[i][2];
    density[i] = rho[i];
    //printf( " drho[i] = %f \n ", drho[i]);
    
  }
    
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSPHTaitwaterMorrisComb::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  memory->create(rho0, n + 1, "pair:rho0");
  memory->create(soundspeed, n + 1, "pair:soundspeed");
  memory->create(B, n + 1, "pair:B");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(viscosity, n + 1, n + 1, "pair:viscosity");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairSPHTaitwaterMorrisComb::settings(int narg, char **arg) {
  if (narg != 0)
    error->all(FLERR,
        "Illegal number of setting arguments for pair_style sph/taitwater/morris");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSPHTaitwaterMorrisComb::coeff(int narg, char **arg) {
  if (narg != 6)
    error->all(FLERR,
        "Incorrect args for pair_style sph/taitwater/morris coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(FLERR,arg[0], atom->ntypes, ilo, ihi);
  force->bounds(FLERR,arg[1], atom->ntypes, jlo, jhi);

  double rho0_one = force->numeric(FLERR,arg[2]);
  double soundspeed_one = force->numeric(FLERR,arg[3]);
  double viscosity_one = force->numeric(FLERR,arg[4]);
  double cut_one = force->numeric(FLERR,arg[5]);
  double B_one = soundspeed_one * soundspeed_one * rho0_one / 7.0;

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    rho0[i] = rho0_one;
    soundspeed[i] = soundspeed_one;
    B[i] = B_one;
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      viscosity[i][j] = viscosity_one;
      //printf("setting cut[%d][%d] = %f\n", i, j, cut_one);
      cut[i][j] = cut_one;

      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSPHTaitwaterMorrisComb::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
    error->all(FLERR,"Not all pair sph/taitwater/morris coeffs are not set");
  }

  cut[j][i] = cut[i][j];
  viscosity[j][i] = viscosity[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHTaitwaterMorrisComb::single(int i, int j, int itype, int jtype,
    double rsq, double factor_coul, double factor_lj, double &fforce) {
  fforce = 0.0;

  return 0.0;
}
