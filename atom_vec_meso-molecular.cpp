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
#include <string.h>
#include <stdlib.h>
#include "atom_vec_meso-molecular.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecMesoMolecular::AtomVecMesoMolecular(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = 0;
  mass_type = 1;
  forceclearflag = 1;

  comm_x_only = 0; // we communicate not only x forward but also vest ...
  comm_f_only = 0; // we also communicate de and drho in reverse direction
  size_forward = 9; // 3 + rho + e + molecule + vest[3], that means we may only communicate 5 in hybrid
  size_reverse = 6; // 3 + drho + de + molecule 
  size_border = 13; // 6 + rho + e + molecule + vest[3] + cv
  size_velocity = 3;
  size_data_atom = 9;
  size_data_vel = 4;
  xcol_data = 7;

  atom->e_flag = 1;
  atom->rho_flag = 1;
  atom->cv_flag = 1;
  atom->vest_flag = 1;
  atom->molecule_flag = 1;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
   ------------------------------------------------------------------------- */

void AtomVecMesoMolecular::grow(int n)
{
  if (n == 0) grow_nmax();
  else nmax = n;
  atom->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  tag = memory->grow(atom->tag, nmax, "atom:tag");
  type = memory->grow(atom->type, nmax, "atom:type");
  mask = memory->grow(atom->mask, nmax, "atom:mask");
  image = memory->grow(atom->image, nmax, "atom:image");
  x = memory->grow(atom->x, nmax, 3, "atom:x");
  v = memory->grow(atom->v, nmax, 3, "atom:v");
  f = memory->grow(atom->f, nmax*comm->nthreads, 3, "atom:f");

  molecule = memory->grow(atom->molecule,nmax,"atom:molecule");

  rho = memory->grow(atom->rho, nmax, "atom:rho");
  drho = memory->grow(atom->drho, nmax*comm->nthreads, "atom:drho");
  e = memory->grow(atom->e, nmax, "atom:e");
  de = memory->grow(atom->de, nmax*comm->nthreads, "atom:de");
  vest = memory->grow(atom->vest, nmax, 3, "atom:vest");
  cv = memory->grow(atom->cv, nmax, "atom:cv");

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
   ------------------------------------------------------------------------- */

void AtomVecMesoMolecular::grow_reset() {
  tag = atom->tag;
  molecule = atom->molecule;
  type = atom->type;
  mask = atom->mask;
  image = atom->image;
  x = atom->x;
  v = atom->v;
  f = atom->f;
  rho = atom->rho;
  drho = atom->drho;
  e = atom->e;
  de = atom->de;
  vest = atom->vest;
  cv = atom->cv;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMolecular::copy(int i, int j, int delflag) {
  //printf("in AtomVecMesoMolecular::copy\n");
  tag[j] = tag[i];
  type[j] = type[i];
  mask[j] = mask[i];
  image[j] = image[i];
  x[j][0] = x[i][0];
  x[j][1] = x[i][1];
  x[j][2] = x[i][2];
  v[j][0] = v[i][0];
  v[j][1] = v[i][1];
  v[j][2] = v[i][2];

  molecule[j] = molecule[i];

  rho[j] = rho[i];
  drho[j] = drho[i];
  e[j] = e[i];
  de[j] = de[i];
  cv[j] = cv[i];
  vest[j][0] = vest[i][0];
  vest[j][1] = vest[i][1];
  vest[j][2] = vest[i][2];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->copy_arrays(i, j,delflag);
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMolecular::force_clear(int n, size_t nbytes)
{
  memset(&de[n],0,nbytes);
  memset(&drho[n],0,nbytes);
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_comm_hybrid(int n, int *list, double *buf) {
  //printf("in AtomVecMesoMolecular::pack_comm_hybrid\n");
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = rho[j];
    buf[m++] = e[j];
    buf[m++] = vest[j][0];
    buf[m++] = vest[j][1];
    buf[m++] = vest[j][2];
    buf[m++] = ubuf(molecule[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::unpack_comm_hybrid(int n, int first, double *buf) {
  //printf("in AtomVecMesoMolecular::unpack_comm_hybrid\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    rho[i] = buf[m++];
    e[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    molecule[i] = (tagint) ubuf(buf[m++]).i;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_border_hybrid(int n, int *list, double *buf) {
  //printf("in AtomVecMesoMolecular::pack_border_hybrid\n");
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = rho[j];
    buf[m++] = e[j];
    buf[m++] = vest[j][0];
    buf[m++] = vest[j][1];
    buf[m++] = vest[j][2];
    buf[m++] = ubuf(molecule[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::unpack_border_hybrid(int n, int first, double *buf) {
  //printf("in AtomVecMesoMolecular::unpack_border_hybrid\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    rho[i] = buf[m++];
    e[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    molecule[i] = (tagint) ubuf(buf[m++]).i;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_reverse_hybrid(int n, int first, double *buf) {
  //printf("in AtomVecMesoMolecular::pack_reverse_hybrid\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = drho[i];
    buf[m++] = de[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::unpack_reverse_hybrid(int n, int *list, double *buf) {
  //printf("in AtomVecMesoMolecular::unpack_reverse_hybrid\n");
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    drho[j] += buf[m++];
    de[j] += buf[m++];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_comm(int n, int *list, double *buf, int pbc_flag,
                           int *pbc) {
  //printf("in AtomVecMesoMolecular::pack_comm\n");
  int i, j, m;
  double dx, dy, dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
      dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
      dz = pbc[2] * domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_comm_vel(int n, int *list, double *buf, int pbc_flag,
                               int *pbc) {
  //printf("in AtomVecMesoMolecular::pack_comm_vel\n");
  int i, j, m;
  double dx, dy, dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
      dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
      dz = pbc[2] * domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMolecular::unpack_comm(int n, int first, double *buf) {
  //printf("in AtomVecMesoMolecular::unpack_comm\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    rho[i] = buf[m++];
    e[i] = buf[m++];
    molecule[i] = (tagint) ubuf(buf[m++]).i;
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMolecular::unpack_comm_vel(int n, int first, double *buf) {
  //printf("in AtomVecMesoMolecular::unpack_comm_vel\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    rho[i] = buf[m++];
    e[i] = buf[m++];
    molecule[i] = (tagint) ubuf(buf[m++]).i;
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_reverse(int n, int first, double *buf) {
  //printf("in AtomVecMesoMolecular::pack_reverse\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
    buf[m++] = drho[i];
    buf[m++] = de[i];
    buf[m++] = ubuf(molecule[i]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMolecular::unpack_reverse(int n, int *list, double *buf) {
  //printf("in AtomVecMesoMolecular::unpack_reverse\n");
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
    drho[j] += buf[m++];
    de[j] += buf[m++];
    molecule[j] = (tagint) ubuf(buf[m++]).i;
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_border(int n, int *list, double *buf, int pbc_flag,
                             int *pbc) {
  //printf("in AtomVecMesoMolecular::pack_border\n");
  int i, j, m;
  double dx, dy, dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = cv[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = cv[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_border_vel(int n, int *list, double *buf, int pbc_flag,
                                 int *pbc)
{
  int i,j,m;
  double dx,dy,dz,dvx,dvy,dvz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = cv[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        buf[m++] = ubuf(molecule[j]).d;
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
        buf[m++] = rho[j];
        buf[m++] = e[j];
        buf[m++] = cv[j];
        buf[m++] = vest[j][0];
        buf[m++] = vest[j][1];
        buf[m++] = vest[j][2];
      }
    } else {
      dvx = pbc[0] * h_rate[0] + pbc[5] * h_rate[5] + pbc[4] * h_rate[4];
      dvy = pbc[1] * h_rate[1] + pbc[3] * h_rate[3];
      dvz = pbc[2] * h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        buf[m++] = ubuf(molecule[j]).d;
        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
          buf[m++] = vest[j][0] + dvx;
          buf[m++] = vest[j][1] + dvy;
          buf[m++] = vest[j][2] + dvz;
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
          buf[m++] = vest[j][0];
          buf[m++] = vest[j][1];
          buf[m++] = vest[j][2];
        }
        buf[m++] = rho[j];
        buf[m++] = e[j];
        buf[m++] = cv[j];
      }
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMolecular::unpack_border(int n, int first, double *buf) {
  //printf("in AtomVecMesoMolecular::unpack_border\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax)
      grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    molecule[i] = (tagint) ubuf(buf[m++]).i;
    rho[i] = buf[m++];
    e[i] = buf[m++];
    cv[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMolecular::unpack_border_vel(int n, int first, double *buf) {
  //printf("in AtomVecMesoMolecular::unpack_border_vel\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax)
      grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    molecule[i] = (tagint) ubuf(buf[m++]).i;
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    rho[i] = buf[m++];
    e[i] = buf[m++];
    cv[i] = buf[m++];
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ----------------------------------------------------------------------
   pack data for atom I for sending to another proc
   xyz must be 1st 3 values, so comm::exchange() can test on them
   ------------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_exchange(int i, double *buf) {
  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  buf[m++] = ubuf(molecule[i]).d;
  buf[m++] = rho[i];
  buf[m++] = e[i];
  buf[m++] = cv[i];
  buf[m++] = vest[i][0];
  buf[m++] = vest[i][1];
  buf[m++] = vest[i][2];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i, &buf[m]);

  buf[0] = m;
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMolecular::unpack_exchange(double *buf) {
  int nlocal = atom->nlocal;
  if (nlocal == nmax)
    grow(0);

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;
  molecule[nlocal] = (tagint) ubuf(buf[m++]).i;
  rho[nlocal] = buf[m++];
  e[nlocal] = buf[m++];
  cv[nlocal] = buf[m++];
  vest[nlocal][0] = buf[m++];
  vest[nlocal][1] = buf[m++];
  vest[nlocal][2] = buf[m++];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]-> unpack_exchange(nlocal,
                                                                   &buf[m]);

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
   ------------------------------------------------------------------------- */

int AtomVecMesoMolecular::size_restart() {
  int i;

  int nlocal = atom->nlocal;
  int n = 18 * nlocal; // 11 + rho + e + cv + vest[3]

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      for (i = 0; i < nlocal; i++)
        n += modify->fix[atom->extra_restart[iextra]]->size_restart(i);

  return n;
}

/* ----------------------------------------------------------------------
   pack atom I's data for restart file including extra quantities
   xyz must be 1st 3 values, so that read_restart can test on them
   molecular types may be negative, but write as positive
   ------------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_restart(int i, double *buf) {
  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];
  buf[m++] = ubuf(molecule[i]).d;
  buf[m++] = rho[i];
  buf[m++] = e[i];
  buf[m++] = cv[i];
  buf[m++] = vest[i][0];
  buf[m++] = vest[i][1];
  buf[m++] = vest[i][2];

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      m += modify->fix[atom->extra_restart[iextra]]->pack_restart(i, &buf[m]);

  buf[0] = m;
  return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
   ------------------------------------------------------------------------- */

int AtomVecMesoMolecular::unpack_restart(double *buf) {
  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    grow(0);
    if (atom->nextra_store)
      memory->grow(atom->extra, nmax, atom->nextra_store, "atom:extra");
  }

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];
  molecule[nlocal] = (tagint) ubuf(buf[m++]).i;
  rho[nlocal] = buf[m++];
  e[nlocal] = buf[m++];
  cv[nlocal] = buf[m++];
  vest[nlocal][0] = buf[m++];
  vest[nlocal][1] = buf[m++];
  vest[nlocal][2] = buf[m++];

  double **extra = atom->extra;
  if (atom->nextra_store) {
    int size = static_cast<int> (buf[0]) - m;
    for (int i = 0; i < size; i++)
      extra[nlocal][i] = buf[m++];
  }

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   create one atom of itype at coord
   set other values to defaults
   ------------------------------------------------------------------------- */

void AtomVecMesoMolecular::create_atom(int itype, double *coord) {
  int nlocal = atom->nlocal;
  if (nlocal == nmax)
    grow(0);

  tag[nlocal] = 0;
  type[nlocal] = itype;
  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];
  mask[nlocal] = 1;
  image[nlocal] = ((imageint) IMGMAX << IMG2BITS) |
    ((imageint) IMGMAX << IMGBITS) | IMGMAX;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;
  molecule[nlocal] = 0;
  rho[nlocal] = 0.0;
  e[nlocal] = 0.0;
  cv[nlocal] = 1.0;
  vest[nlocal][0] = 0.0;
  vest[nlocal][1] = 0.0;
  vest[nlocal][2] = 0.0;
  de[nlocal] = 0.0;
  drho[nlocal] = 0.0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
   ------------------------------------------------------------------------- */

void AtomVecMesoMolecular::data_atom(double *coord, imageint imagetmp, char **values) {
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = ATOTAGINT(values[0]);
  molecule[nlocal] = ATOTAGINT(values[1]);
  type[nlocal] = atoi(values[2]);
  if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
    error->one(FLERR,"Invalid atom type in Atoms section of data file");

  rho[nlocal] = atof(values[3]);
  e[nlocal] = atof(values[4]);
  cv[nlocal] = atof(values[5]);

  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];

  image[nlocal] = imagetmp;

  mask[nlocal] = 1;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  vest[nlocal][0] = 0.0;
  vest[nlocal][1] = 0.0;
  vest[nlocal][2] = 0.0;

  de[nlocal] = 0.0;
  drho[nlocal] = 0.0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack hybrid quantities from one line in Atoms section of data file
   initialize other atom quantities for this sub-style
   ------------------------------------------------------------------------- */

int AtomVecMesoMolecular::data_atom_hybrid(int nlocal, char **values) {

  molecule[nlocal] = ATOTAGINT(values[0]);
  rho[nlocal] = atof(values[1]);
  e[nlocal] = atof(values[2]);
  cv[nlocal] = atof(values[3]);

  return 4;
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecMesoMolecular::pack_data(double **buf)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = ubuf(tag[i]).d;
    buf[i][1] = ubuf(molecule[i]).d;
    buf[i][2] = ubuf(type[i]).d;
    buf[i][3] = rho[i];
    buf[i][4] = e[i];
    buf[i][5] = cv[i];
    buf[i][6] = x[i][0];
    buf[i][7] = x[i][1];
    buf[i][8] = x[i][2];
  }
}

/* ----------------------------------------------------------------------
   pack hybrid atom info for data file
------------------------------------------------------------------------- */

int AtomVecMesoMolecular::pack_data_hybrid(int i, double *buf)
{
  buf[0] = ubuf(molecule[i]).d;
  buf[1] = rho[i];
  buf[2] = e[i];
  buf[3] = cv[i];
  return 4;
}

/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecMesoMolecular::write_data(FILE *fp, int n, double **buf)
{
  for (int i = 0; i < n; i++)
    fprintf(fp,TAGINT_FORMAT
            " %d %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e \n",
            (tagint) ubuf(buf[i][0]).i,
            (tagint) ubuf(buf[i][1]).i,
            (int) ubuf(buf[i][2]).i,
            buf[i][3],buf[i][4],buf[i][5],
            buf[i][6],buf[i][7],buf[i][8]
  );
}

/* ----------------------------------------------------------------------
   write hybrid atom info to data file
------------------------------------------------------------------------- */

int AtomVecMesoMolecular::write_data_hybrid(FILE *fp, double *buf)
{
  fprintf(fp,TAGINT_FORMAT " %-1.16e %-1.16e %-1.16e %-1.16e",
    ubuf(buf[0]).i,buf[1],buf[2],buf[3]);
  return 4;
}

/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecMesoMolecular::property_atom(char *name)
{
  if (strcmp(name,"rho") == 0) return 0;
  if (strcmp(name,"drho") == 0) return 1;
  if (strcmp(name,"e") == 0) return 2;
  if (strcmp(name,"de") == 0) return 3;
  if (strcmp(name,"cv") == 0) return 4;
  if (strcmp(name,"mol") == 0) return 5;
  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecMesoMolecular::pack_property_atom(int index, double *buf,
                                     int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

  if (index == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = rho[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 1) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = drho[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 2) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = e[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 3) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = de[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 4) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = cv[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 5) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = ubuf(molecule[i]).d;
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
   ------------------------------------------------------------------------- */

bigint AtomVecMesoMolecular::memory_usage() {
  bigint bytes = 0;

  if (atom->memcheck("tag"))
    bytes += memory->usage(tag, nmax);
  if (atom->memcheck("type"))
    bytes += memory->usage(type, nmax);
  if (atom->memcheck("mask"))
    bytes += memory->usage(mask, nmax);
  if (atom->memcheck("image"))
    bytes += memory->usage(image, nmax);
  if (atom->memcheck("x"))
    bytes += memory->usage(x, nmax, 3);
  if (atom->memcheck("v"))
    bytes += memory->usage(v, nmax, 3);
  if (atom->memcheck("f"))
    bytes += memory->usage(f, nmax*comm->nthreads, 3);
  if (atom->memcheck("molecule")) 
    bytes += memory->usage(molecule,nmax);
  if (atom->memcheck("rho"))
    bytes += memory->usage(rho, nmax);
  if (atom->memcheck("drho"))
    bytes += memory->usage(drho, nmax*comm->nthreads);
  if (atom->memcheck("e"))
    bytes += memory->usage(e, nmax);
  if (atom->memcheck("de"))
    bytes += memory->usage(de, nmax*comm->nthreads);
  if (atom->memcheck("cv"))
    bytes += memory->usage(cv, nmax);
  if (atom->memcheck("vest"))
    bytes += memory->usage(vest, nmax);

  return bytes;
}
