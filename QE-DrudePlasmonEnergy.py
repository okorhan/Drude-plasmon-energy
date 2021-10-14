#!/bin/python3
"""
The code is a post-processing tool for evaluating the approximate Fermi velocity, and
the approximate Drude plasmon energy of the metallic systems from the XML file produced by
'scf' or 'nscf' simulation on a fine uniform grid using the Quantum Espresso software.

In this version, the Brillouin zone sampling is restricted to the Monkhorst-Pack equivalent uniform gird with
or without symmetry operations imposed.

It was written by Okan K. Orhan on June 2020 during the Covid-19 quarantine days.

The code has to be run in the same directory of the Quantum Espresso simulation by:

python3 QE-DrudePlasmonEnergy.py Sample.in Sample.out

The format of Sample.in

& DrudePlasmonEnergy
    Prefix= String                              --> Same prefix with the QE run
    Outdir= String                     --> Same outdir with the QE run
    DOSFermi = Real                       --> Density of states at the Fermi level in 1/eV unit
    FermiSurfaceThickness = Real          --> Slap thickness of the Fermi plates
    FermiSurfaceInterpolation = Integer   --> Brillouin zone grid for interpolation
    QPFermiShift = 0.0                    --> Fermi level shift after applied QP stretching operators
    QPValStretching= 1.0                  --> Valence band stretching operator to approximate QP bands
    QPCondStretching= 1.0                 --> Conduction band stretching operator to approximate QP bands

Input tips:

1 ) Set the QP input parameters to their default values above and use KS-DFT DOS at Fermi level to calculate Drude
    plasmon energy for the KS-DFT bands.

2 ) Use FermiSurfaceInterpolation = Integer*(k_point_number + 1) if you want to ensure the original k-points
    are included during after interpolaton.
     

"""




# User-define functions

def BrillouinZoneCompletion(SubOut):

    # Collecting the reciprocal cell vectors

    Bmat = []
    trcell = SubOut.find('basis_set').find('reciprocal_lattice')
    for tk in trcell:
        Bmat.append(np.array([float(i) for i in tk.text.split()]))
    Bmat = np.array(Bmat)
    del trcell, tk

    # Transformation matrix from between the cartesian (K) and the crystal (H) coordinates
    # in the reciprocal space

    TKHmat = np.linalg.inv(np.transpose(Bmat))
    THKmat = np.linalg.inv(TKHmat)

    # Collecting the reduced k-points given in the crystal coord.

    ReducedKPointsH = []

    for tkout in SubOut.iter('k_point'):
        tk = np.array([float(i) for i in tkout.text.split()])
        ReducedKPointsH.append(TKHmat.dot(tk))
    del tkout

    # Collecting the symmetry operations

    CrySymH = []

    for tsym in SubOut.iter('symmetry'):
        tmat1 = tsym.find('rotation').text.split()
        tmat2 = np.array([float(tmat1[i]) for i in range(9)])
        tmat3 = np.transpose(np.array([tmat2[0:3], tmat2[3:6], tmat2[6:9]]))
        CrySymH.append(tmat3)
    del tmat1, tmat2, tmat3

    # Constructing the full Brillouin zone

    FullKPointsH = []
    FullKPointsIndex = []
    kind = 0
    for tk in ReducedKPointsH:
        ltk = []
        for tsym in CrySymH:
            stk = tsym.dot(tk)
            for i in range(3):
                if stk[i] >= 0.5:
                    stk[i] -= 1.0
                if stk[i] < - 0.5:
                    stk[i] += 1.0
            ltk.append(stk)
        for iltk in np.unique(ltk, axis=0):
            FullKPointsH.append(iltk)
            FullKPointsIndex.append(kind)
        kind += 1

    for i in range(len(FullKPointsH)):
        tk = FullKPointsH[i]
        tind = FullKPointsIndex[i]
        if tk[0] == -0.5:
            ttk = tk + np.array([1.0, 0.0, 0.0])
            FullKPointsH.append(ttk)
            FullKPointsIndex.append(tind)
        if tk[1] == -0.5:
            ttk = tk + np.array([0.0, 1.0, 0.0])
            FullKPointsH.append(ttk)
            FullKPointsIndex.append(tind)
        if tk[2] == -0.5:
            ttk = tk + np.array([0.0, 0.0, 1.0])
            FullKPointsH.append(ttk)
            FullKPointsIndex.append(tind)
        if tk[0] == -0.5 and tk[1] == -0.5:
            ttk = tk + np.array([1.0, 1.0, 0.0])
            FullKPointsH.append(ttk)
            FullKPointsIndex.append(tind)
        if tk[0] == -0.5 and tk[2] == -0.5:
            ttk = tk + np.array([1.0, 0.0, 1.0])
            FullKPointsH.append(ttk)
            FullKPointsIndex.append(tind)
        if tk[1] == -0.5 and tk[2] == -0.5:
            ttk = tk + np.array([0.0, 1.0, 1.0])
            FullKPointsH.append(ttk)
            FullKPointsIndex.append(tind)
        if tk[0] == -0.5 and tk[1] == -0.5 and tk[2] == -0.5:
            ttk = tk + np.array([1.0, 1.0, 1.0])
            FullKPointsH.append(ttk)
            FullKPointsIndex.append(tind)

    del ReducedKPointsH

    return np.array(Bmat), np.array(TKHmat), np.array(THKmat), np.array(FullKPointsH), np.array(FullKPointsIndex)

def FermiSmearedScissor(Ei,Ef,kT,Sv,Sc):
    FermiFunc=1.0/(math.exp((Ei-Ef)/kT)+1)
    QPStretch=FermiFunc*Sv+(1-FermiFunc)*Sc
    return QPStretch


import numpy as np
import math, sys
import xml.etree.ElementTree as et
from scipy.interpolate import Rbf
#import matplotlib.pyplot as plt

# Global constants
Ha2eV=27.211396132

kT=0.1 # eV



# Global variables
InpKeys=['Prefix', 'Outdir', 'DOSFermi', 'FermiSurfaceThickness','FermiSurfaceInterpolation',\
         'QPFermiShift', 'QPValStretching', 'QPCondStretching']




# Arguments for executable
fexe, fileinp, fileout=sys.argv

fout=open(fileout,"w")
fout.write('######################################################################\n'
           '#                                                                    #\n'
           '#   Drude plasmon energy calculated by  QE-DrudePlasmonEnergy.py     #\n'
           '#   written by Okan K. Orhan on June 2020 during the Coivd19 days    #\n'
           '#                                                                    #\n'
           '######################################################################\n\n')

# Reading the input file

InpLines=[]
fout.write('\n Checking the input file...')
with open(fileinp) as fp:
    if fp.readline().strip().split(" ")[1] != 'DrudePlasmonEnergy':
        fout.write('\n ERROR: Input file error! ')
        sys.exit('\n Calculation terminated! ')
    for line in fp.readlines():
        if line.strip():
            InpLines.append(line.strip().split("="))

InpLines=np.array(InpLines)
if len(InpLines) > 8:
    fout.write('\n ERROR: Input file error! ')
    sys.exit('\n Calculation terminated! ')
else:
    for count  in range(8):
        if InpLines[count,0].strip() not in InpKeys:
            print(InpLines[count,0])
            fout.write('\n ERROR: Input file error! ')
            sys.exit('\n Calculation terminated! ')


# Reading the Quantum Espresso XML file

tfile=InpLines[1,1].strip(",").strip("'").strip()+"/"+InpLines[0,1].strip(",").strip("'").strip()\
        +".save/data-file-schema.xml"

while True:
    try:
        with open(tfile, 'rb') as tf:
            if str(tf.readline())[4:7] != 'xml':
                fout.write('\n XML file not found! ')
                sys.exit('\n Calculation terminated! ')
        XMLfile=et.parse(tfile)
        if str(XMLfile.getroot().tag[-8:])!='espresso':
            fout.write('\n ERROR: This file was not generated by the Quantum Espresso! ')
            sys.exit('\n Calculation terminated! ')
        break
    except FileNotFoundError:
        fout.write('\n ERROR: File not found! \n')
        sys.exit('\n Calculation terminated! ')
    except IsADirectoryError:
        fout.write('\n ERROR: This is a directory!\n')
        sys.exit('\n Calculation terminated! ')

while True:
    try:
        DOSFermi=float(InpLines[2,1].strip())
        FermiSurfaceThickness=float(InpLines[3,1].strip())
        FermiSurfaceInterpolation=int(InpLines[4,1].strip())
        QPFermiShift=float(InpLines[5,1].strip())
        QPValStretching=float(InpLines[6,1].strip())
        QPCondStretching=float(InpLines[7,1].strip())
        break
    except ValueError:
        fout.write("\n ERROR: Input value error!")
        sys.exit('\n Calculation terminated! ')

fout.write('\n Extracting information from XML file... ')
SubIn=XMLfile.find('input'); SubOut=XMLfile.find('output')

# Fermi energy from XML file in eV

FermiEnergy=Ha2eV*float(SubOut.find('band_structure').find('fermi_energy').text)
fout.write('\n Fermi energy: %0.5f eV' % (FermiEnergy))

# Checking Brillouin zone sampling

tktype=SubOut.find('band_structure').find('starting_k_points').find('monkhorst_pack')

if tktype.text == 'Monkhorst-Pack':
    tkey=tktype.attrib.keys()
    tknum=str([int(tktype.attrib.get(i)) for i in tkey])
    fout.write('\n Monkhorst-Pack equivalent uniform Brillouin zone: %s' % tknum)
else:
    fout.write("\n ERROR: Not Monkhorst-Pack equivalent uniform Brillouin zone!")
    sys.exit('\n Calculation terminated! ')


# Constructiong full Brillouin zone

alat=float(SubOut.find('atomic_structure').attrib.get('alat'))
KMetric=2.0*math.pi/alat

Bmat, TKHmat, THKmat, FullKPointsH, FullKPointsIndex = BrillouinZoneCompletion(SubOut)

FullKPointsK=[]
for tk in FullKPointsH:
    FullKPointsK.append(THKmat.dot(tk))
FullKPointsK=np.array(FullKPointsK)
iBmat=np.linalg.inv(Bmat)



# Valence electron numbers

NElec=float(SubOut.find('band_structure').find('nelec').text)
NVal=np.array([NElec+NElec*i*0.01 for i in np.linspace(-10,10,21)])
LowBand=math.floor(NVal[0]/2)-1
HighBand=math.ceil(1.3*(NVal[-1]/2))


# Collecting the eigenvalues on the reduced Brillouin zone

ReducedEigen=[]
for teig in SubOut.iter('eigenvalues'):
    te = np.array([Ha2eV*float(i)-FermiEnergy for i in teig.text.split()])
    ReducedEigen.append(te[0:HighBand])
ReducedEigen=np.array(ReducedEigen).T

# Determining the metallic bands for the

MetalBands=[]
for ib in range(len(ReducedEigen)):
    if len(np.unique(np.sign(ReducedEigen[ib]))) > 1:
        MetalBands.append(ib)
MetalBands=np.array(MetalBands)
fout.write('\n Metallic band(s): %s' % str(MetalBands+1))


# Interpolating the metallic bands

esmear=FermiSurfaceThickness
nkp=FermiSurfaceInterpolation
kx=FullKPointsH[:,0]
ky=FullKPointsH[:,1]
kz=FullKPointsH[:,2]
kgrid=np.linspace(np.amin(kx),np.amax(kx),nkp)
dk=0.5*0.1*((np.amax(kx)-np.amin(kx))/nkp)

vtot = np.array([0.0, 0.0, 0.0]);v2totK = 0.0;v2totS = 0.0;Stot = 0.0;nferk = 0

Sw = np.array([0.5, 0.5, 0.5])
Sw = THKmat.dot(Sw)
Sw /= np.linalg.norm(Sw);

for ibn in MetalBands:
    QPBand = []
    KSBand = ReducedEigen[ibn, FullKPointsIndex]
    for ieig in KSBand:
        QPStretch = FermiSmearedScissor(ieig, 0.0, kT, QPValStretching, QPCondStretching)
        QPBand.append(QPStretch * ieig - QPFermiShift)
    QPBand = np.array(QPBand)
    tEint = Rbf(kx, ky, kz, QPBand)

    for i in kgrid:
        for j in kgrid:
            for k in kgrid:
                if abs(tEint(i, j, k)) <= esmear:
                    vi = (tEint(i + dk, j, k) - tEint(i - dk, j, k)) / (2 * dk)
                    vj = (tEint(i, j + dk, k) - tEint(i, j - dk, k)) / (2 * dk)
                    vk = (tEint(i, j, k + dk) - tEint(i, j, k - dk)) / (2 * dk)
                    v = np.array([vi, vj, vk])
                    v = KMetric * THKmat.dot(v)
                    dSw = abs((v / np.linalg.norm(v)) * Sw)
                    vtot += v
                    v2totK += v.dot(v)
                    v2totS += (dSw * v).dot(dSw * v)
                    Stot += np.linalg.norm(dSw)
                    nferk += 1
if nferk == 0:
    nferk = 1
if Stot == 0:
    Stot = 1
fermi = FermiEnergy+QPFermiShift
#vavr = np.linalg.norm(vtot)
vavr = [abs(round(i,3)) for i in vtot]
v2avrK = v2totK / nferk
v2avrS = v2totS / Stot
wpK = math.sqrt((4.0 * math.pi / 3.0) * v2avrK * DOSFermi)
wpS = math.sqrt((4.0 * math.pi / 3.0) * v2avrS * DOSFermi)

fout.write('\n Fermi Energy: %0.5f eV'
           '\n Density of states at Fermi level: %0.5f eV^-1'
           '\n Total number of sampled point on the Fermi surface(s): %d'
           '\n Total area of the Fermi surface(s): %0.5f'
           '\n Average Fermi velocity: %s'
           '\n Expectation value of Fermi velocity square w/o surface weighting: %0.5f'
           '\n Expectation value of Fermi velocity square w surface weighting: %0.5f'
           '\n Drude plasmon energy w/o surface weighting: %0.3f'
           '\n Drude plasmon energy w surface weighting: %0.3f' % (fermi, DOSFermi, nferk, Stot,\
                                                                   str(vavr), v2avrK, v2avrS, \
                                                                   wpK, wpS))



fout.close()









