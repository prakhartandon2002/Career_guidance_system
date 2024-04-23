#!/bin/csh -fb
########################################################################
#   WARNING:  Do NOT Modify This File!                                 #
#                                                                      #
#   This file was generated by PRS.  Modifying it will invalidate the  #
#   PRS results.  If you feel you need to change this file, please     #
#   contact your PRS administrator first.                              #
#                                                                      #
#   You Have Been Warned!                                              #
########################################################################
#
#lsf: bsub -J SRM_THE_FM_S_dcnxt_2_dcg_ex8/CortexM3 -o lsf.output -oo lsf.out -W 240:00 -P "platform=spf:#:product=dcnxt:#:project=nightly" -oo lsf.out -sp 50 -g /Scheduler/project=dc:Sandbox:SRM_THE_FM_S/actual_requestor=prakhart -P '/Scheduler/project=dc:Sandbox:SRM_THE_FM_S:#:actual_requestor=prakhart' -app batch -R "rusage[mem=4G]" -R "select[cpu_code='gold6248']" -R "span[hosts=1]" -n 4  -P "harness=prs:#:identifier=1712315670:#:jobtype=qor:#:platform=spf:#:product=dcnxt:#:project=nightly:#:/Scheduler/project=dc:Sandbox:SRM_THE_FM_S:#:actual_requestor=prakhart:#:suitetype=RMS:#:flowtype=subflow:#:toolfrom=rtlopt:#:toolto=nwprpt:#:design=CortexM3"
#grd: /remote/pv/bin/24x7/qsub -N SRM_THE_FM_S_dcnxt_2_dcg_ex8%CortexM3 -cwd -j y -l arch=glinux,os_bit=64 -o grd.out -r n -js 100 -ac actual_requestor='prakhart' -ac scheduler:project='dc/Sandbox/SRM_THE_FM_S' -l cpu_code=gold6248 -l mem_free=4G -pe mt 4 -P batch 
#

rm -f CortexM3.all.done

# Host Info
/remote/pv/bin/pvwatch  >& CortexM3.hinfo.out

# Exec Info
/u/prsuite/prs/etc/execinfo.pl  CortexM3.all.sum >& CortexM3.einfo.out


# Run rtlopt
./CortexM3.rtlopt.csh >& CortexM3.rtlopt.out

# Run dcopt
./CortexM3.dcopt.csh |& tee CortexM3.dcopt.out

# Run dcrpt
./CortexM3.dcrpt.csh >& CortexM3.dcrpt.out

# Run dccmd
./CortexM3.dccmd.csh >& CortexM3.dccmd.out

# Run nw2nlib
./CortexM3.nw2nlib.csh >& CortexM3.nw2nlib.out

# Run nwpopt
./CortexM3.nwpopt.csh >& CortexM3.nwpopt.out

# Run nwprpt
./CortexM3.nwprpt.csh >& CortexM3.nwprpt.out

# Run fmchk
./CortexM3.fmchk.csh |& tee CortexM3.fmchk.out

# Run fmrpt
./CortexM3.fmrpt.csh >& CortexM3.fmrpt.out

# Run clean
./CortexM3.clean.csh >& CortexM3.clean.out


touch CortexM3.all.done
