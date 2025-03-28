C  NPAGFULLA11.FOR                                         6/30/14

C  NPAGFULLA11 HAS THE FOLLOWING CHANGES TO NPAGFULL11:

C  1. JUST AFTER THE REWIND(27) STATEMENT, SUBROUTINE NEWWORK1 IS 
C  CALLED TO READ THE PATIENT DATA FROM FILE 27, AND CONVERT IT TO
C  PATIENT DATA ON FILE 37, WITH EACH STEADY STATE DOSE INDICATOR
C  RESULTING IN AN EXTRA 100 DOSE SETS (WITH THE NEGATIVE DOSE TIME
C  LEFT IN - SEE COMMENTS BELOW). NOTE THAT THIS SUBROUTINE NEWWORK1
C  IS THE SAME AS IN THE CURRENT NPAG "ENGINE" MODULE, npageng25.f,
C  EXCEPT IT READS FILE 27 AND WRITES TO FILE 37, RATHER THAN READING
C  FROM FILE 23 AND WRITING TO FILE 27; ARRAY TIMOBREL IS NOT
C  ESTABLISHED IN NEWWORK1, AND NOT PASSED AS AN ARGUMENT (IT IS NOT
C  NEEDED IN THIS PROGRAM); MAXSUB IS ALSO NOT PASSED AS AN 
C  ARGUMENT (IT IS NOT NEEDED); NSUB IS NOT PASSED AS AN ARGUMENT (IT
C  IS NOT NEEDED); AND COMMON/DOSEOBS AND THE ARRAYS IN IT ARE DELETED
C  (THEY ARE NOT NEEDED).

C  NOTE THAT SUBROUTINE NEWWORK1 IS INCLUDED IN THE NPAGFULLA MODULE.

C  2. IF THE PROGRAM BOMBS, THE MESSAGE THAT IS WRITTEN TO THE SCREEN
C  WILL NOW ALSO BE WRITTEN TO THE FILE ERRFIL = ERRORxxxx, WHERE xxxx
C  IS THE 4-DIGIT RUN NO. IN THIS WAY, IF THE PROGRAM IS BEING RUN USING
C  Pmetrics, THE Pmetrics PROGRAM CAN RESPOND APPROPRIATELY. NOTE THAT
C  ERRFIL IS PASSED TO ALL THE ROUTINES WHICH COULD WRITE TO IT
C  USING COMMON/ERR/ERRFIL.

C  3. THE MAXIMUM NO. OF OUTPUT EQUATIONS WILL BE CHANGED FROM 6 TO 7,
C  AND TO FACILITATE ANY FUTURE SUCH CHANGES, THIS NUMBER WILL BE SET
C  = MAXNUMEQ (SO ONLY THE PARAMETER STATEMENT WILL HAVE TO BE CHANGED
C  IN THE FUTURE). RATHER THAN PASS MAXNUMEQ TO ALL THE RELEVANT 
C  SUBROUTINES (AS IS DONE IN NPAG113.FOR AND IT2B110.FOR), THIS 
C  PROGRAM WILL JUST HAVE MAXNUMEQ SET IN A PARAMETER STATEMENT IN ALL
C  THESE ROUTINES. AND NOTE THAT IN THOSE SUBROUTINES, ANY 6
C  REFERRING TO THE MAX. NO. OF OUTPUT EQUATIONS WILL BE CHANGED TO
C  MAXNUMEQ.

C  4. COMMON/OBSER IN MAIN IS REMOVED. IT WASN'T NEEDED. SIMILARLY
C  ALL THE ARRAYS IN THIS COMMON ARE NO LONGER DIMENSIONED.

C  5. FORMATS 2345 AND 2346 NOW CORRECTLY REFER TO SUBROUTINE
C  NPAGFULL1, RATHER THAN NPAGFULL.

C-----------------------------------------------------------------------

C  NPAGFULL11.FOR                                       8/06/13

C  NPAGFULL11 HAS THE FOLLOWING CHANGE2 FROM NPAGFULL1:
C  1. INSTEAD OF SELELCTING ALL GRID PTS. WHOSE PROBALITIES ARE WITHIN
C  1.D-10 OF THE HIGHEST, THIS PROGRAM TAKES ALL POINTS WHOSE PROBS ARE
C  WITHIN 1.D-100 OF THE HIGHEST.

C  SEE THE "IF(MAXCYC .EQ. 0) THEN" CODE NEAR THE END OF MAIN.

C  2. THE MAIN SUBROUTINE IS RENAMED NPAGFULL11 (INSTEAD OF NPAGFULL).

C  3. ALL SUBROUTINES OTHER THAN THE MAIN ONE ARE REMOVED SINCE THEY
C  ARE ALREADY IN NPAGFULL.FOR, WHICH WILL ALSO BE LINKED TO
C  BESTDOSTEMP.FOR (TO BE RENAMED BESTDOS112.FOR)

C  4. MAXCYC IS REMOVED FROM THE ARGUMENT LIST SINCE IT IS HARDCODED
C  TO BE 0 BELOW.

C----------------------------------------------------------------------- 

C  NPAGFULL1.FOR                                        6/30/13

C  NPAGFULL1 HAS THE FOLLOWING CHANGES TO NPAGFULL:

C - MAXCYC IS MANUALLY SET = 0, SO THE VALUE INPUT IN THE ARGUMENT 
C   STATEMENT IS UNIMPORTANT. 
C - AFTER LOOP 800, CONTROL GOES TO LABEL 900 IF MAXCYC = 0. SO THIS 
C   IS SIMILAR CODE TO npageng24.f, IN THAT IF MAXCYC = 0, NO CYCLE 
C   CALCS. ARE DONE --> SUBROUTINE emint IS NEVER CALLED.
C - AT LABEL 900, IF MAXCYC = 0 (WHICH OF COURSE IT IS), THEN THE
C   REMAINING CODE (FROM SUBROUTINE SUBRES OF npageng24.f) IS USED
C   TO FINISH CALCULATING THE BAYESIAN POSTERIOR OF THE SUBJECT.
C   AND THIS NEW CORDEN IS RETURNED TO THE MAIN BESTDOS MODULE.

C------------------------------------------------------------------

C  NPAGFULL.FOR                                            3/26/13

C  NPAGFULL IS BASED ON THE npageng22.f PROGRAM. IT RUNS AN NPAG
C  ANALYSIS IN ORDER TO OBTAIN THE FULL POSTERIOR DENSITY OF A SUBJECT
C  GIVEN AN APRIORI DENSITY. ALL OTHER CODE IN npageng22.f IS 
C  REMOVED (E.G., ALL EXTRA CALCULATIONS, ALL WRITING TO FILES, ETC.).

C  NOTE THAT ALL INFO NEEDED BY THIS ROUTINE IS INCLUDED IN THE 
C  CALLING ARGUMENTS; IN PARTICULAR, npag102.inp IS NOT READ.

C  THIS COMPARES TO NPAGBAY, WHICH CALCULATED THE 0-CYCLE BAYESIAN 
C  POSTERIOR OF THE SUBJECT.

C  NOTE ALSO THAT ALL DIMENSIONS OF 500 RELATED TO DOSE EVENTS HAVE BEEN
C  CHANGE TO 5000.

C----------------------------------------------------------------------- 

c  npageng22.f                                             11/8/12

c  npageng22 has the following change from npageng21:

c  1. It comments out the PAUSE statement following Format 164 in 
c  Subroutine emint. Reason: the program will not complete properly if 
c  it is run under  Pmetrics (which cannot supply a keyboard response
c  during a run). 

c  2. Formats 1657 and 7124 are changed to show that the output file
c  is made by npageng22 rather than npageng21. 

c-----------------------------------------------------------------------

c  See npageng22.f code for all the comments from npageng21.f back
c  to m2_5calf.f.


C-----------------------------------------------------------------------
 
C***********************************************************************

      SUBROUTINE NPAGFULL11(MAXSUB,MAXGRD,MAXDIM,NVAR,NUMEQT,WORK,WORKK,
     1 CORDEN,NDIM,MF,RTOL,ATOL,NOFIX,IRAN,VALFIX,AB,ierrmod,GAMLAM0,
     2 NGRID,NACTVE,PYJGX,DENSTOR,CORDLAST)

C  NOTE FOR NPAGFULL11, MAXCYC IS NOT SUPPLIED FROM BESTDOS... SINCE
C  IT WILL BE HARDCODED TO BE 0 BELOW.


      IMPLICIT REAL*8(A-H,O-Z)

      PARAMETER(MAXNUMEQ=7)


      DIMENSION WORK(MAXGRD),WORKK(MAXGRD),PYJGX(MAXSUB,MAXGRD),
     1 DENSTOR(MAXGRD,4),CORDEN(MAXGRD,MAXDIM+1),
     2 CORDLAST(MAXGRD,MAXDIM+1), YO(594,MAXNUMEQ),SIG(594,MAXNUMEQ),
     3 AB(30,2),X(30),VALFIX(20),IRAN(32),PX(32),ATOL(20),
     4 C0(MAXNUMEQ),C1(MAXNUMEQ),C2(MAXNUMEQ),C3(MAXNUMEQ),ATOLL(20)

C  NOTE THAT ALL DIMENSIONS = 150 HAVE BEEN CHANGED TO 594, SINCE THIS
C  NO. REPRESENTS THE TOTAL NO. OF OBSERVATIONS (AND THE MAX. NO IS
C  6 OUTPUT EQUATIONS x 99 OBSERVATIONS/EQ). THIS COULD BE CHANGED
C  TO NUMEQT*MAXOBS, BUT IT WOULD BE MORE TROUBLE THAN IT'S WORTH TO
C  MAKE THESE DIMENSIONS VARIABLE.

      CHARACTER ERRFIL*20

      COMMON/ERROR/ERRFIL
C  COMMON/ERROR/ IS SUPPLIED TO ALL THE ROUTINES WHICH COULD WRITE TO
C  ERRFIL. AND ERRFIL IS MADE CHARACTER *20 ABOVE.

 
	COMMON SIG
   	COMMON/TOUSER/NDIMM,MFF,RTOLL,ATOLL
	COMMON/NXER/NXE
C NXE FROM ABOVE COMMON IS NO. OF TIMES XERRWD IS CALLED.

C  THE BLANK COMMON ABOVE IS SUPPLIED TO SUBROUTINE IDPC.


C  COMMON/TOUSER IS SUPPLIED TO SUBROUTINE USERANAL IN idfixed.f.
C  COMMON/OBSER/ IS SUPPLIED FROM SUBROUTINE FILREAD.
C  NO! AS OF NPAGFULLA.FOR, COMMON/OBSER IS REMOVED FROM MAIN. IT IS
C  UNNEEDED. SIMILARLY, TIMOB, DOSTIM, RS, Y00, AND BS ARE NO LONGER
C  DIMENSIONED IN MAIN.


C***********************************************************************
 
C-----------------------------------------------------------------------
 
    2 FORMAT(A20)
  222 FORMAT(A3)
 2222 FORMAT(A5)
 
C-----------------------------------------------------------------------
 
C  SET MAXCYC = 0
      MAXCYC = 0

      NSUB = 1

      NDIMM = NDIM
      MFF = MF
      RTOLL = RTOL
      DO I = 1,NDIM
       ATOLL(I) = ATOL(I)
      END DO

C  THE ABOVE VALUES HAD TO BE ESTABLISHED SINCE THE SAME VARIABLES
C  CANNOT BE IN COMMON STATEMENTS IF THEY ARE DUMMY CALLING ARGUMENTS.
 
      


C  NOTE THAT THIS SUBROUTINE WAS CALLED BY BESTDOSxxx.FOR/MAIN, WHICH
C  HAS ALREADY WRITTEN TO SCRATCH FILE 27 THE DATA FROM THE SINGLE
C  SUBJECT WHOSE POSTERIOR DENSTIY IS TO BE CALCULATED, BASED ON THE
C  PRIOR DENSITY INPUT PASSED TO THIS ROUTINE IN CORDEN.


c  As of npageng14.f, tol is hardcoded to be 1.D-4. Previously, it
c  was allowed to be any positive number .GE. 1.D-4. 

        tol = 1.D-4


C  ESTABLISH ASSAY VALUES FROM ierrmod AND gamlam0.


	gamma = 1.d0
	flat = 1.d0
	if(ierrmod .eq. 2) gamma = gamlam0
        if(ierrmod .eq. 3) gamma = gamlam0
        if(ierrmod .eq. 4) flat = gamlam0

	igamma = 0
        gamdel=0.1
        if(ierrmod.eq.1) gamdel=0.d0


C  CHANGE NGRID TO BE MAXGRD, IF IT IS > MAXGRD.

C???DEBUG 3/23. NGRID IS NOT A DIMENSION --> IT IS NEEDED TO BE
C  WHATEVER IT IS FROM THE NPAGDENFILE INPUT INTO THE BESTDOS MAIN
C  MODULE. SO DON'T LIMIT IT TO BE .LE. MAXGRD.

C        if(ngrid .gt. MAXGRD) then
 
C          write(6,*)
 
C          write(6,*) 'requested NGRID = ',NGRID 
C          write(6,*) 'maximum allowable is MAXGRD = ',MAXGRD
C          write(6,*) 'resetting NGRID = ',MAXGRD
C          write(6,*) 'to fit in available storage'
C          write(6,*)
 
C          ngrid = MAXGRD
 
C         endif



C  CALCULATE VOLSPA, THE 'VOLUME' OF THE INTEGRATION SPACE (NEEDED IN
C  CALLS TO NOTINT).
 
      VOLSPA=1.D0
      DO 170 I=1,NVAR
  170 VOLSPA = VOLSPA*(AB(I,2)-AB(I,1))
 
 
C  NOTE IN THIS PROGRAM, THE USER WILL ALWAYS INPUT A PRIOR DENSITY
C  WITH VALUES IN CORDEN, SO THE ICYCLE = 0 CODE HAS BEEN REMOVED.	

 
C  AS OF npageng19.f, PRESET NACTLAST TO BE NACTVE. THIS WAY, IN THE
C  UNLIKELY EVENT THAT THE FIRST CYCLE OF A RUN HAS A HESSIAN ERROR
C  (WHICH MEANS THAT WHEN CONTROL COMES BACK TO MAIN FROM SUBROUTINE
C  emint, IT IS TRANSFERRED TO LABEL 900 AND THEREFORE SKIPS THE
C  cbegin statistics  SECTION WHERE NACTLAST = NACTVE IS SET), THERE
C  WON'T BE A PROBLEM WHEN NACTVE IS SET = NACTLAST JUST BELOW LABEL 
C  900. IN PREVIOUS PROGRAMS, IN THE ABOVE SITUATION, BELOW LABEL 900,
C  NACTVE = NACTLAST WOULD RESULT IN NACTVE BEING SET = 0 SINCE 
C  NACTLAST WAS UNITIALIZED.

      NACTLAST = NACTVE 


        prefobj=-1.d30
        prebig=-1.d30

 
C  SET ICYCLE = 0. THE PROGRAM WILL RUN UP TO MAXCYC CYCLES.

      ICYCLE = 0

C  CORDEN HOLDS, IN ITS FIRST NACTVE ROWS, THE STARTING JOINT DENSITY
C  AND COORDINATE VALUES. FOR K=1,NACTVE, CORDEN(K,J) = JTH COORDINATE
C  OF THE KTH ACTIVE POINT, J=1,NVAR; AND CORDEN(K,NVAR+1) IS THE
C  ASSOCIATED DENSITY FOR THE KTH ACTIVE POINT.
 
C  IF ICYCLE .GT. 0, CORDEN WAS READ IN.
C  IF ICYCLE = 0, NACTVE=NGRID, AND CORDEN WAS FILLED AT LABEL 30 ABOVE.
C		 IN THIS CASE, THE DENSITY IS UNIFORM, SO ALL
C		 CORDEN(K,NVAR+1) VALUES = 1/VOLSPA, K=1,NACTVE.
 
 
C	IPRED=11 + ICYCLE
C	JCOL=0
	ITEST=0
 
C  IPRED IS THE CYCLE NO. WHERE THE NEXT 2-CYCLE PREDICTION
C  ALGORITHM STARTS (IT IS NO LONGER USED). JCOL = COLUMN NO. OF DENSTOR
C  IN WHICH IS STORED THE DENSITY OF ONE OF THE 2-CYCLES USED IN THE
C  PREDICTION (IT IS NO LONGER USED). IT IS SET = 0 ABOVE, SINCE NO 
C  STORAGE IS REQUIRED UNTIL CYCLE NO. 11


C  (SEE BELOW). ITEST=0 --> THE NEXT CYCLE IS NOT (INITIALIZED) TO BE
C  A TEST CYCLE (SEE CODE BELOW WHEN ITEST=1,2, OR 3).
 
C    NEW FOR m2_13cal.f: NSTORE SET = 0. NSTORE IS THE NO. OF GRID
C    POINTS, WHOSE P(YJ|X) VALUES HAVE BEEN STORED IN PYJGX IN LOOP 800.
C    THIS NO. CAN BE CHANGED BY THE 'CONDENSING' CODE BELOW, SINCE
C    INACTIVE POINTS ARE THROWN OUT.
 
	NSTORE=0
cadapt    initialize grid resoution to 20%
          resolve=0.20



 
 1001 ICYCLE=ICYCLE+1


cgam3
10001 continue
c above is new entry point for gammaplus/minus eps tries
      itest = 0
 
 
cadapt  reset number of stored points to that before expansion
c     nstore=nstoresv
 

C 1239   FORMAT(///' CYCLE NO.',I5,/)

C  ICYCLE IS THE NUMBER OF THE NEXT CYCLE TO BE RUN.
C
C  THIS IS WHERE EACH NEW CYCLE STARTS (FOR EACH CYCLE, THE DENSITY OF
C  X IS UPDATED FROM THE PREVIOUS DENSITY ESTIMATE, USING THE
C  OBSERVED SUBJECT DATA FROM THE INPUT DATA FILES WHICH ARE PASSED TO
C  SUBROUTINE IDPC BELOW.
C

 
C  START THE SUBJECT LOOP.
 

C  REWIND SCRATCH FILE 27 WHICH HAS ALL THE SUBJECT DATA FILES
C  CONCATENATED ON IT, IN ORDER. ACTUALLY THERE IS ONLY NSUB = 1
C  SUBJECT IN THIS RUN.
 
	REWIND(27)

C  PUT THE SUBJECT'S DATA ONTO FILE 37 BY CALLING SUBROUTINE NEWWORK1
C  (SIMILAR ROUTINE TO THE ONE IN npageng25.f, EXCEPT IT READS FILE 27
C  AND WRITES TO FILE 37, RATHER THAN READING FROM FILE 23 AND WRITING
C  TO FILE 27). NOTE THAT IT HAS NO ARGUMENTS; NONE ARE NEEDED IN THIS
C  PROGRAM.

C  NOTE THAT IF THE SUBJECT FILE HAS NO STEADY STATE DOSE INDICATORS, IT
C  WILL NOT BE CHANGED; IF IT DOES, IT WILL BE ALTERED TO INCLUDE AN
C  EXTRA 100 DOSES SETS FOR EACH STEADY STATE DOSE INDICATOR. NOTE THAT
C  SUBROUTINE NEWWORK1 WILL LEAVE IN THE NEGATIVE DOSE TIME (WHICH IS
C  THE STEADY STATE DOSE INDICATOR) BECAUSE THE ID ROUTINES NEED TO SEE
C  THIS INDICATOR TO KNOW THAT A STEADY STATE DOSE SET IS COMING.

       OPEN(37)


       CALL NEWWORK1
       REWIND(37)
 



C  NOBTOT WILL BE THE RUNNING TOTAL OF ALL NON-MISSING OBSERVED VALUES
C  OVER ALL THE NSUB SUBJECTS. THIS IS NEEDED TO CALCULATE BIC BELOW.

        NOBTOT = 0


      DO 1000 JSUB=1,NSUB


C  CALL SUBROUTINE FILREAD TO READ, FOR THIS SUBJECT, FROM SCRATCH FILE
C  37, THE NO. OF OBSERVATION TIMES (NOBSER) AS WELL AS THE
C  OBSERVED VALUES THEMSELVES: YO(I,J) = THE 'NOISY' OBSERVED VALUES
C  FOR THIS SUBJECT; I=1,NOBSER, J=1,NUMEQT. THESE OBSERVED VALUES ARE
C  USED ONLY TO CALCULATE THE ASSAY STANDARD DEVIATIONS (USING THE
C  VECTORS, C0,C1,C2,C3, WHICH ARE ALSO READ IN). THE REST OF THE INFO
C  IN THE SUBJECT DATA FILE IS PASSED IN COMMONS TO THE IDPC MODULE
C  SUBROUTINES.
 
	CALL FILREAD(NOBSER,YO,C0,C1,C2,C3)

C  FIND THE ASSAY STANDARD DEVIATIONS FOR THIS SUBJECT. FOR EACH
C  OF THE NOBSER*NUMEQT OBSERVED VALUES (EXCEPT THAT YO(I,J) = -99 -->
C  OUTPUT EQ. J HAS NO OBSERVED LEVEL FOR OBSERVATION TIME I),
C  Y, SIG = C0 + C1*Y + C2*Y**2 + C3*Y**3.
C  NOTE THAT, THEORETICALLY, SIG SHOULD BE A CUBIC FNT. OF
C  THE 'TRUE' OBSERVED VALUES, NOT THE 'NOISY' OBSERVED VALUES (BUT THE
C  'TRUE' VALUES ARE UNKNOWN).
 
C  ALSO, CALCULATE SIGFAC, THE PRODUCT OF THE NON-MISSING STD. DEV.'S
C  (A NON-MISSING S.D. IS ONE FOR WHICH THE CORRESPONDING YO(I,J) IS
C  .NE. -99, THE MISSING VALUE CODE).
C  INITIALIZE SIGFAC=1, AND THEN UPDATE IT FOR EACH NON-MISSING
C  OBSERVATION.
 
C  MISVAL WILL BE THE RUNNING TOTAL OF MISSING VALUES AMONG ALL THE
C  NUMEQT x NOBSER POTENTIAL OBSERVED LEVELS.

	MISVAL = 0
 
        SIGFAC=1.D0
 
 	DO 140 I=1,NOBSER
 	 DO 140 J=1,NUMEQT
 
	  Y = YO(I,J)
 
C  IF Y = -99, IT MEANS THAT OUTPUT EQ. J HAD NO VALUE AT OBSERVATION
C  TIME I. IN THIS CASE, IGNORE THIS Y AND INCREASE MISVAL BY 1.
 

	  IF(Y .EQ. -99) THEN
	   MISVAL = MISVAL+1
	   GO TO 140
	  ENDIF
 
C  NOTE: FOR EACH SUBJECT, MUST ENSURE THAT ALL THE STD DEV'S ARE NON-
C        ZERO. OTHERWISE, THE PROGRAM WILL BLOW UP! THIS IS BECAUSE
C        P(YJ|X) INVOLVES SQUARED DIFFERNCES BETWEEN OBSERVED Y'S AND
C        EXPECTED Y'S (FOR EACH X GRID POINT)...EACH DIFFERENCE
C        NORMALIZED (I.E., DIVIDED) BY THE VARIANCE OF THE RESPECTED
C        OBSERSATION.
 
C 	 SEE M2_17CAL.F CODE FOR COMMENTS ON HOW A STD. DEV. COULD = 0.
 
C  ALSO TEST TO MAKE SURE NO STD. DEV. < 0, SINCE SIGFAC BEING NEGATIVE
C  WOULD RESULT IN A NEGATIVE PROBABILITY (SEE PYJGX CALCULATION BELOW).
 
      SIG(I,J) = C0(J)+C1(J)*Y+C2(J)*Y*Y+C3(J)*Y**3
cgam4
      if(ierrmod.eq.2) sig(i,j) = sig(i,j)*gamma
      if(ierrmod.eq.3) sig(i,j)=dsqrt(sig(i,j)**2 + gamma**2)
      if(ierrmod.eq.4) sig(i,j) = gamma*flat

      IF(SIG(I,J) .EQ. 0) THEN

		WRITE(*,2345) JSUB
2345        FORMAT(//' A S.D. IS 0 FOR JSUB = ',I5,'. RERUN THE '/
     1' PROGRAM WITH C0 NOT = 0  FOR THIS SUBJECT, OR WITH THIS'/
     2' SUBJECT ELIMINATED.'//
     3' THIS IS IN SUBROUTINE NPAGFULL11.'/)
	   	CLOSE(37)

        OPEN(47,FILE=ERRFIL)
         WRITE(47,2345) JSUB    
        CLOSE(47)


		CALL PAUSE
	   	STOP

      ENDIF

      IF(SIG(I,J) .LT. 0) THEN

		WRITE(*,2346) JSUB
2346            FORMAT(//' A S.D. < 0 FOR JSUB = ',I5,'. RERUN THE '/
     1' PROGRAM WITH A BETTER CHOICE FOR THE ASSAY ERROR POLYNOMIAL'/
     2' COEFFICIENTS.'//
     3' THIS IS IN SUBROUTINE NPAGFULL11.'/)
	   	CLOSE(37)

        OPEN(47,FILE=ERRFIL)
         WRITE(47,2346) JSUB    
        CLOSE(47)

		CALL PAUSE
	   	STOP

      ENDIF
 
      SIGFAC=SIGFAC*SIG(I,J)
 
  140 CONTINUE
 
C  NOTE THAT SIGFAC WAS CALCULATED IN LOOP 140 ABOVE, AND THAT OFAC IS
C  NOW THE RESULT OF (NOBSER*NUMEQT - MISVAL) VALUES.
 
        OFAC=2.506628274631**(NOBSER*NUMEQT - MISVAL)
        NOBTOT = NOBTOT + NOBSER*NUMEQT - MISVAL

 
C  NOTE THAT 2.5066... = SQRT(2*PI).
 
C  FOR EACH SUBJECT, AND EACH GRID POINT, CALL IDPC, A SUBROUTINIZED
C  VERSION OF THE ADAPT PROGRAM ID3 TO CALCULATE THE SUM OF SQUARES OF
C  DIFFERENCES BETWEEN THE OBSERVED VALUES AND PREDICTED (BY THE MODEL)
C  VALUES (NORMALIZED BY THE ASSAY VARIANCE OF EACH OBSERVATION) ...

 8888   FORMAT(' ',' CYCLE ',I5,',  SUBJECT ',I5,' ...  % COMPLETED = ',
     1F8.2)
	XNEXT = 1.D0

C  SEVERAL CHANGES FOR m2_13cal.f ARE IN LOOP 800.
 
	DO 800 IG=1,NACTVE
 

C  PRINT TO THE SCREEN THE UPDATE ON WHAT % OF GRID POINTS HAVE BEEN
C  CALCULATED IF NACTVE > NSTORE (I.E., IF NACTVE .LE. NSTORE -->
C  ALL P(YJ|X)'s ARE ALREADY STORED INTO PYJGX AND SO THIS 8OO LOOP
C  WILL GO VERY FAST.
 
	IF(NACTVE .GT. NSTORE) THEN	
 
C  PRINT GRID PT. AND % COMPLETED TO SCREEN.
	XPER=IG*100.D0/NACTVE

	IF(XPER .GE. XNEXT) THEN

        IF(ICYCLE.eq.1) THEN
	 WRITE(*,8888) ICYCLE,JSUB,XPER
	 IF(NXE .GT. 0) WRITE(*,1254) NXE
 1254    FORMAT('  TOTAL NO. OF NUM. INTEG. WARNINGS IS ',I20)
	ENDIF

	 XNEXT=XNEXT+1.D0
 
	ENDIF

	ENDIF 
 
	IF(IG .LE. NSTORE) GO TO 700
 
	
C  ESTABLISH THE IGTH GRID POINT. IT IS STORED IN ROW IG OF
C  CORDEN.
 
	DO J=1,NVAR
	 X(J)=CORDEN(IG,J)
	END DO
 
C  ESTABLISH THE COMBINED RANDOM AND FIXED PARAMETER VALUES INTO
C  PX -- IN THE CORRECT ORDER AS INDICATED BY VECTOR IRAN. CALL
C  MAKEVEC TO DO THIS.

 
	CALL MAKEVEC(NVAR,NOFIX,IRAN,X,VALFIX,PX)

	CALL IDPC(PX,W)
 
C  W RETURNS AS THE SUM OF:
C  ((YO(I,J)-H(I,J))/SIG(I,J))**2, WHERE H(I,J) = PREDICTED VALUE OF THE
C  JTH OUTPUT EQ AT THE ITH OBSERVATION TIME, ASSUMING THE IGTH GRID
C  POINT, X, ... OVER THE NOBSER x NUMEQT QUANTITIES ABOVE WHICH DON'T
C  HAVE YO(I,J) = -99 (WHICH MEANS THAT OUTPUT EQ. J HAS NO OBSERVED
C  LEVEL FOR TIME I).
 
C  CALCULATE P(YJ|X) FOR X-GRID POINT NO. IG.
 
C  THIS NEXT TEST IS FOR THE PC. AS AN EXAMPLE, THE COMPAC COMPUTER
C  CANNOT HANDLE ARGUMENTS TO DEXP WHICH ARE SMALLER THAN -11354. SINCE
C  THE ARGUMENT TO DEXP BELOW IS -.5*W, SET PYJGX = 0 IF W IS .GT.
C  22708.
 
C  SEE CODE AFTER CALCULATION OF P(YJ) TO SEE WHAT HAPPENS IF ALL THE
C  P(YJ|X) ARE  SET = 0.
 
C  NOTE THAT WORKK WILL ALWAYS BE SET = P(YJ|X=IG GRID PT), WHICH IS
C  NEEDED IN THE CALCULATION OF DXI (NOTE DXI NOT USED AS OF 
C  bignpaglap1.f) SINCE PYJGX WILL NOT BE COMPLETE IF NACTVE > MAXGRD.

	IF(IG .LE. MAXGRD) PYJGX(JSUB,IG)=0.D0
	WORKK(IG) = 0.D0
 
	IF(W .LE. 22708.D0) THEN
	 IF(IG .LE. MAXGRD) PYJGX(JSUB,IG) = DEXP(-.5D0*W)/SIGFAC/OFAC
	 WORKK(IG) = DEXP(-.5D0*W)/SIGFAC/OFAC
	ENDIF
 
C  CALCULATE P(X,YJ) FOR X-GRID POINT NO. IG. PUT IT INTO WORK(IG).
 
	IF(IG .GT. MAXGRD) THEN
	 WORK(IG) = WORKK(IG)*CORDEN(IG,NVAR+1)
	 GO TO 800
	ENDIF
 
  700   WORK(IG)=PYJGX(JSUB,IG)*CORDEN(IG,NVAR+1)

	WORKK(IG) = PYJGX(JSUB,IG)

C???DEBUG
C	WRITE(*,3631) ICYCLE,IG,PYJGX(JSUB,IG)
C 3631 FORMAT(' ICYCLE,IG,PYJGX(JSUB,IG): ',I3,2X,I2,2X,F20.15)
 	
  800   CONTINUE

C???DEBUG
C	CALL PAUSE
	
 
C  CALCULATE P(YJ), A SCALAR WHICH IS THE INTEGRAL OF P(X,YJ) OVER

C  X-SPACE.
 
C  CALL NOTINT, AN INTEGRATION ROUTINE. THE
C  FOLLOWING IS SUPPLIED TO THIS ROUTINE:
C  VOLSPA = VOLUMNE OF THE INTEGRATION SPACE.
C  NGRID = NO. OF ORIGINAL GRID POINTS.
C  NACTVE = NO. OF ACTIVE GRID POINTS.
C  WORK(I), I=1,NACTVE = VALUE OF THE FUNCTION TO BE INTEGRATED, AT
C                       THE ITH GRID POINT.
C  MAXGRD  = THE DIMENSION OF WORK.
  
      CALL NOTINT(VOLSPA,NGRID,NACTVE,WORK,MAXGRD,PYJ)
 
 
C  IF PYJ RETURNS AS 0, IT IS BECAUSE P(X,YJ)=WORK IS 0 IN ALL ITS
C  NACTVE ENTRIES. THIS OCCURS WHEN EACH OF NACTVE VALUES OF W (WHICH
C  RETURNS FROM THE CALLS TO IDPC) IS LARGER THAN 1416 (SINCE P(YJ|X)
C  INVOLVES e RAISED TO THE POWER -.5*W, AND e RAISED TO A POWER
C  SMALLER THAN -708 IS SET TO 0 BY, FOR EXAMPLE, THE COMPAC COMPUTER).
C

C  IN CASE THIS HAPPENS, PRINT A MESSAGE TO THE USER AND STOP.
C
	IF (PYJ .EQ. 0.D0) THEN

		WRITE(*,26)
   26       FORMAT(//' FOR THIS SUBJECT, THE PROB. OF THE OBSERVED'/
     1' CONCENTRATIONS (FOR THE INDICATED DOSAGE REGIMEN), GIVEN EACH '/
     2' AND EVERY GRID POINT IN THE ESTABLISHED GRID, IS 0. THE '/
     3' PROGRAM STOPS. THE USER SHOULD CONSIDER INCREASING THE'/
     4' NO. OF GRID POINTS ALLOWED (HARDCODED INTO MAIN), AND/OR '/
     5' NARROWING THE GRID BOUNDARIES OF THE VARIABLES, AND/OR  '/
     6' INCREASING THE SIZES OF (C0,C1,C2,C3), THE ASSAY NOISE '/
     7' COEFFICIENTS. ALL OF THESE CHANGES WILL HAVE THE EFFECT OF'/
     8' MAKING SOME OF THE ABOVE CONDITIONAL PROBABILITES LARGER.'//
     9' THIS IS IN SUBROUTINE NPAGFUL.'/)

        OPEN(47,FILE=ERRFIL)
         WRITE(47,26)     
        CLOSE(47)

		CALL PAUSE
		STOP

	ENDIF


C??? FOR NPAGFULL1.FOR, IF MAXCYC = 0, GO TO LABEL 900, WHERE
C  THE CODE WILL BE TO FINISH CALCULATING THE SIMPLE BAYESIAN
C  POSTERIOR FOR THIS SUBJECT, BASED ON THE ABOVE NACTVE GRID PTS.

      IF(MAXCYC .EQ. 0) GO TO 900


  
 1000   CONTINUE


c begin optimization
cgam5
cgam5 - from here (immediately after 1000   CONTINUE to
cgam5 - immediately before c end optimization was lifted
cgam5 - from gamadapt1.f, replacing old material beteen these limits
      igamma = igamma + 1
      if(ierrmod.eq.1) igamma=1
csdsc - added April 2, 2000
c con first iteration, call hte interior point method

      if(mod(igamma,3).eq.1) then

      gammab = gamma
      gammap = gamma * (1.d0+gamdel)
      gammam = gamma / (1.d0+gamdel)
      call emint(pyjgx,maxsub,corden,maxgrd,nactve,nsub,1,
     &corden(1,nvar+1),denstor(1,1),denstor(1,2),denstor(1,3),
     &fobj,gap,nvar,keep,IHESS)

C  AS OF npageng18.f, IHESS IS ADDED TO ARGUMENT LIST OF emint.
C  IF IHESS RETURNS AS -1, IT MEANS THE HESSIAN MATRIX IN THE INTERIOR
C  POINTS ALGORITHM WAS SINGULAR. IN THIS CASE, GO TO LABEL 900 TO
C  CREATE THE OUTPUT FILES BASED ON THE VALUES FROM THE PREVIOUS CYCLE.
C  NOTE THAT BY GOING TO LABEL 900, NOTHING FROM THIS CYCLE WILL BE
C  WRITTEN TO FILE 25, AND THE CORDEN(.,.) MATRIX WILL REMAIN WHAT IT
C  WAS FROM THE PREVIOUS CYCLE.

      IF(IHESS .EQ. -1) GO TO 900

      nactve = keep

C???DEBUG
C	WRITE(*,*)' AFTER CALL EMINT, NO. 1, NACTVE = KEEP = ',NACTVE


      call emint(pyjgx,maxsub,corden,maxgrd,nactve,nsub,0,
     &corden(1,nvar+1),denstor(1,1),denstor(1,2),denstor(1,3),
     &fobj,gap,nvar,keep,IHESS)

C???DEBUG
C	WRITE(*,*)' AFTER CALL EMINT, NO. 2, NACTVE = ',NACTVE



C  AS OF npageng18.f, IHESS IS ADDED TO ARGUMENT LIST OF emint.

C  IF IHESS RETURNS AS -1, IT MEANS THE HESSIAN MATRIX IN THE INTERIOR
C  POINTS ALGORITHM WAS SINGULAR. IN THIS CASE, GO TO LABEL 900 TO
C  CREATE THE OUTPUT FILES BASED ON THE VALUES FROM THE PREVIOUS CYCLE.
C  NOTE THAT BY GOING TO LABEL 900, NOTHING FROM THIS CYCLE WILL BE
C  WRITTEN TO FILE 25, AND THE CORDEN(.,.) MATRIX WILL REMAIN WHAT IT
C  WAS FROM THE PREVIOUS CYCLE.

      IF(IHESS .EQ. -1) GO TO 900


      fobjbase = fobj


      nactve0 = nactve
c new on Jan 2, 2002 - save otpimal solution in denstor(1,4)
c so that stat program can work on best of base, up, and down
c solutions
      do i=1,nactve
      denstor(i,4)=corden(i,nvar+1)
      enddo
      nstore = 0
      fobjbest = fobjbase

      if(ierrmod.eq.1) go to 14001
      gamma = gammap
      go to 10001

      endif
cgamma above endif is for mod(igamma,3).eq.1 case

      if(mod(igamma,3).eq.2) then

      call emint(pyjgx,maxsub,corden,maxgrd,nactve,nsub,0,
     &corden(1,nvar+1),denstor(1,1),denstor(1,2),denstor(1,3),
     &fobj,gap,nvar,keep,IHESS)

C???DEBUG
C	WRITE(*,*)' AFTER CALL EMINT, NO. 3, NACTVE = ',NACTVE



C  AS OF npageng18.f, IHESS IS ADDED TO ARGUMENT LIST OF emint.
C  IF IHESS RETURNS AS -1, IT MEANS THE HESSIAN MATRIX IN THE INTERIOR
C  POINTS ALGORITHM WAS SINGULAR. IN THIS CASE, GO TO LABEL 900 TO
C  CREATE THE OUTPUT FILES BASED ON THE VALUES FROM THE PREVIOUS CYCLE.
C  NOTE THAT BY GOING TO LABEL 900, NOTHING FROM THIS CYCLE WILL BE
C  WRITTEN TO FILE 25, AND THE CORDEN(.,.) MATRIX WILL REMAIN WHAT IT
C  WAS FROM THE PREVIOUS CYCLE.

      IF(IHESS .EQ. -1) GO TO 900
      fobjplus = fobj

c new  Jan 2, 2002 - save solution if fobjplus is better than fobjbase
      if(fobjplus.gt.fobjbest) then
      fobjbest = fobjplus
      do i=1,nactve

      denstor(i,4) = corden(i,nvar+1)
      enddo
      endif
      gamma = gammam

         go to 10001

      endif

      if(mod(igamma,3).eq.0) then

      call emint(pyjgx,maxsub,corden,maxgrd,nactve,nsub,0,
     &corden(1,nvar+1),denstor(1,1),denstor(1,2),denstor(1,3),
     &fobj,gap,nvar,keep,IHESS)

C???DEBUG
C	WRITE(*,*)' AFTER CALL EMINT, NO. 4, NACTVE = ',NACTVE




C  AS OF npageng18.f, IHESS IS ADDED TO ARGUMENT LIST OF emint.

C  IF IHESS RETURNS AS -1, IT MEANS THE HESSIAN MATRIX IN THE INTERIOR
C  POINTS ALGORITHM WAS SINGULAR. IN THIS CASE, GO TO LABEL 900 TO
C  CREATE THE OUTPUT FILES BASED ON THE VALUES FROM THE PREVIOUS CYCLE.
C  NOTE THAT BY GOING TO LABEL 900, NOTHING FROM THIS CYCLE WILL BE
C  WRITTEN TO FILE 25, AND THE CORDEN(.,.) MATRIX WILL REMAIN WHAT IT
C  WAS FROM THE PREVIOUS CYCLE.

      IF(IHESS .EQ. -1) GO TO 900

      fobjminu = fobj

      if(fobjminu.gt.fobjbest) then
      fobjbest = fobjminu
      do i=1,nactve
      denstor(i,4) = corden(i,nvar+1)
      enddo
      endif

      endif

cgamma - above statement changed from "nstore = nactve" to force
c reevaluation of all points.
      
c now temporairily reset to gamma
      gamma = gammab
      fobj = fobjbase
      if(fobjplus.gt.fobjbase) then
      gamma = gammap

      fobj = fobjplus
      gamdel = 4.*gamdel
      endif

      if(fobjminu.gt.fobjbase) then
      gamma = gammam
      fobj = fobjminu
      gamdel = 4.*gamdel
      endif
      gamdel = gamdel*0.5
      if(gamdel.lt.0.01) gamdel=0.01
14001 continue
cgam5 above label is entry point for ierrmod = 1 (no  gamma) case

c corden(*,nvar+1) sums to 1 when it comes out of emint
c Now reset forden(i,nvar+1) to best of three solutions
c and normalize to funny BIGNPEM factor
      fact=ngrid/volspa
      do i=1,nactve
      corden(i,nvar+1)=fact*denstor(i,4)
      enddo

C???DEBUG
C	WRITE(*,*)' AT END OF OPTIMIZATION, NACTVE = ',NACTVE
C	CALL PAUSE


cend optimization


cbegin statistics


c now we compute all hte statistical stuff using this distribution
c and the full nactve (before condensation) points.
c Later, in the condensation performed just before the grid refienment
c and subsequent expansion, we will condense by just using the
c 'keep' flags in DENSTOR(i,1) that emint left there.  The density will
c not be updated to refelct this cahnge (there is no need)
c until the next call to emint


c  As of npageng18.f, save CORDEN to CORDLAST AND NACTVE TO NACTLAST.
c  The reason is that if, somewhere during the next cycle's calculations
c  (during one of the calls to Subroutine emint), a Hessian Matrix is
c  singular, then IHESS will be set = -1, and the program will stop.
c  And in this case, the program must be able to write out all of the

c  information from this cycle (the last completed cycle). And that
c  means that the CORDEN from this cycle (which will be stored into 
c  CORDLAST), and NACTVE (store into NACTLAST) should be used in the
c  call to Subroutine SUBRES in loop 7000. Otherwise, the CORDEN and
c  NACTVE used in that call would have already partly updated in the
c  next cycle before the Hessian error occurred.

      DO I = 1,NACTVE
       DO J = 1,NVAR+1
       CORDLAST(I,J) = CORDEN(I,J)
       END DO
      END DO

C???DEBUG
C	WRITE(*,*)' CORDEN IS NOW '
C	DO I = 1,NACTVE
C	 WRITE(*,5497) (CORDEN(I,J),J=1,3)
C 5497  FORMAT(3(F12.8,2X))
C	END DO
C	CALL PAUSE

      NACTLAST = NACTVE


	IF(MAXCYC .EQ. 0) GO TO 900

C  Starting with bigmlt1.f, this is a jump point.


cend statistics
cbegin control
c we are now done wtih statistics - this is the best place to
c check for whether we can exit - if so , last printed statistic
c will agree with current density corden, and corden is still
c correct (e.g. after condensation-expansion, it is no longer
c correct until we call emint again)
cint.9 control section to check for exit criteria, resolution 
c  refinement, and end of major cycles

cint9.a  first, we exit if we have reached maxcyc on cycle counter

C  SET IMAXCYC = 0; IF IT CHANGES TO 1, IT MEANS THAT MAXCYC CYCLES
C  HAVE BEEN RUN, AND THE PROGRAM WILL STOP.

        IMAXCYC = 0

      if(icycle .ge. maxcyc) then



C  SET IMAXCYC = 1 --> MAXCYC WAS REACHED.

        IMAXCYC = 1

C  COMMENT OUT THE GO TO 900 STATEMENT BELOW SINCE EVEN IF ICYCLE
C  = MAXCYC, THE PROGRAM STILL NEEDS TO TEST TO SEE IF CONVERGENCE
C  WAS ACHIEVED IN THE FINAL CYCLE.
C      go to 900

      endif

c  The above endif is for the  if(icycle .ge. maxcyc)  condition.


cint9.b  second, we check improvement from last cycle

      ximprove=fobj-prefobj
      prefobj = fobj

cint9.c if ximprove is too low, refine the resolve criterion

      if(dabs(ximprove) .le. tol .and. resolve .gt. 0.0001) then
        resolve=resolve*0.5
      endif

cint9.d check to see if resolve bottoms out - if so, start a new
c major cycle by resetting it to its highest allowable value, or
c exit if the improvment from the last major cycle is too small ...

C  AND EXIT IF IMAXCYC = 1 (SEE ABOVE; THIS MEANS THAT THE MAX. NO.
C  OF CYCLES HAS ALREADY BEEN RUN AND THE ONLY REASON THIS PART OF THE
C  CODE IS BEING RUN IS TO SEE IF CONVERGENCE WAS ACHIEVED IN THE FINAL
C  CYCLE.

      if(resolve.le.0.0001) then


c      saveres = resolve
        resolve=0.2
        checkbig = fobj - prebig
        prebig =fobj

C  NOTE THAT THE 
C  CONVERGENCE CRITERION IS THAT DABS(CHECKBIG) .LE. .01. 


C	 WRITE(*,1023) ICYCLE
C1023    FORMAT(/' FOR CYCLE NO, ',I6,' THE CONVERGENCE CRITERION AND ME
C     1DIANS ARE: ')

C	 WRITE(*,1024) DABS(checkbig)

C1024    FORMAT(1X,G14.4,' <-- CONVERGENCE OCCURS WHEN THIS NO. < .01')


        if(dabs(checkbig) .le. 0.01) then

C  CONVERGENCE HAS BEEN ACHIEVED.


          go to 900

        endif

      endif

c  above endif is for the  if(resolve .le. .0001)  condition.


C  IF IMAXCYC = 1, THE MAX. NO. OF CYCLES HAVE ALREADY BEEN RUN -->
C  GO TO 900. THE ONLY REASON THIS PART OF THE CODE WAS BEING RUN IS TO
C  SEE IF CONVERGENCE WAS ACHIEVED IN THIS FINAL CYCLE, AND THAT WAS
C  JUST TESTED ABOVE.
 
        IF(IMAXCYC .EQ. 1) GO TO 900


cend control
cbegin expansion

         nactveold=nactve

         do ipoint=1,nactveold
c first, divide current probability into 2*nvar+1 pieces
           pcur=corden(ipoint,nvar+1)/(2*nvar+1)
c update original point
           corden(ipoint,nvar+1)=pcur
             do ivar=1,nvar
	       del=(ab(ivar,2)-ab(ivar,1))*resolve
c create first new trial point at -eps in coordinate ivar
               do i=1,nvar
                  corden(nactve+1,i)=corden(ipoint,i)
	       enddo
	       corden(nactve+1,ivar)=corden(nactve+1,ivar)-del
               corden(nactve+1,nvar+1)=pcur
               ntry=nactve+1
c icheck that new point is at least minimally distant from old points

               call checkd(corden,ntry,nactve,ab,maxgrd,nvar,iclose)
c only keep trial lower point if it lies above lower bound and satisfies
c minimal distance requirement
	       if(corden(nactve+1,ivar).ge.ab(ivar,1)) then
                if(iclose.eq.0) nactve=nactve+1
	       endif
c now create second trail point at +eps in coordinate ivar
               do i=1,nvar
                 corden(nactve+1,i)=corden(ipoint,i)
	       enddo
	       corden(nactve+1,ivar)=corden(nactve+1,ivar)+del
               corden(nactve+1,nvar+1)=pcur
c only keep upper point if it lies below upper bound and
c satisfies distance requirement
               ntry=nactve+1
               call checkd(corden,ntry,nactve,ab,maxgrd,nvar,iclose)
	       if(corden(nactve+1,ivar).le.ab(ivar,2)) then
	         if(iclose.eq.0) nactve=nactve+1
               endif
             enddo
c    above enddo for loop over ivar=1,nvar

           enddo
c    above enddo for loop over ipoint=1,nactveold


cend expansion
c go to begin new cycle

        prefobj=fobj



        GO TO 1001
 
  900 continue


C??? NEW CODE BELOW FOR NPAGFULL1.FOR.
      IF(MAXCYC .EQ. 0) THEN

C  NOTE THAT IF MAXCYC = 0, CONTROL HAS BEEN TRANSFERRED HERE AFTER
C  LOOP 800 ABOVE. SO FINISH CALCULATING THE BAYESIAN POSTERIOR.

C  THE BAYESIAN POSTERIOR DENSITY OF THIS SUBJECT IS, FOR GRID PT. IG,

C  P(XIG|YJ) = P(YJ,XIG)/P(YJ). PUT THESE VALUES INTO CORDEN(IG,NVAR+1).

	DO IG=1,NACTVE
	 CORDEN(IG,NVAR+1) = WORK(IG)/PYJ
	END DO

C  FOR NPAGFULL11.FOR, THE 1.D-10 BELOW IS CHANGED TO 1.D-100.

C  CALCULATE HOW MANY OF THE NACTVE GRID POINTS FROM THE FINAL CYCLE
C  ARE "ACTIVE" (WITHIN 1.D-10 OF THE MAXIMUM DENSITY FOR THIS SUBJECT).
C  ... AND, AS OF npageng23.f, ELIMINATE NON-SIGNIFICANT GRID PTS. IN
C  CORDEN (PREVIOUSLY ALL THE POINT FROM THE FINAL CYCLE DENSITY
C  SHOWED UP IN CORDEN, EVEN THOSE WITH INSIGNIFICANT PROBABILITIES).
C  AND NOTE THAT THE BAYESIAN POSTERIOR DENSITY FOR THIS SUBJECT
C  WILL BE STORED INTO BAYPOS(JSUB,.,.), AND PASSED IN COMMON/BAY
C  TO SUBROUTINE READOUT. AND NACTSUB(JSUB) WILL CONTAIN THE NO. OF
C  ACTIVE GRID POINTS FOR THIS SUBJECT'S BAYESIAN POSTERIOR DENSITY.

C  AND NOTE THAT THE NOMINAL DIMENSIONS OF BAYPOS, (800,1500,31),
C  CANNOT BE EXCEEDED BECAUSE THESE ARE THE VALUES FOR MAXSUB, MAXGRD,
C  AND MAXDIM+1, AS SPECIFIED IN THE PARAMETER STATEMENT IN THE PC PREP
C  MAIN MODULE (CURRENTLY NPAG111.FOR). BUT NOTE THAT npageng23.f WILL
C  NOT EXECTUTE WITH THESE DIMENSIONS BECAUSE IT IS TOO BIG FOR A 
C  WIN32 APPLICATION (SEE NPAG111.EXP). SO, IN THIS ROUTINE, AND IN
C  SUBROUTINE READOUT (IN read19.f), THE FIRST DIMENSION HAS BEEN 
C  REDUCED TO 100 (AND SIMILARLY FOR THE DIMENSION OF NACTSUB). IF
C  JSUB > 100, THE BAYESIAN POSTERIOR VALUES BELOW WILL NOT BE STORED.

        DENMAX=CORDEN(1,NVAR+1)

        DO I=1,NACTVE
         D=CORDEN(I,NVAR+1)
         IF(D .GT. DENMAX) DENMAX=D
        END DO
 
        SUMD = 0.D0
        NEWIND = 0

        DO I=1,NACTVE
         D=CORDEN(I,NVAR+1)
          IF(D .GT. 1.D-100*DENMAX) THEN
          SUMD=SUMD+D
          NEWIND=NEWIND+1
           DO J=1,NVAR
            CORDEN(NEWIND,J) = CORDEN(I,J)
           END DO
          CORDEN(NEWIND,NVAR+1)=D
         ENDIF
        END DO

        NACTVE = NEWIND

        FACT = NGRID/VOLSPA/SUMD
 
	DO I=1,NACTVE
	  CORDEN(I,NVAR+1) = CORDEN(I,NVAR+1)*FACT
	END DO


      ENDIF
C  THE ABOVE ENDIF IS FOR THE  IF(MAXCYC .EQ. 0)  CONDITION.



C  AS OF npageng18.f, CONTROL CAN BE TRANSFERRED TO LABEL 900 DIRECTLY
C  AFTER RETURNING FROM A CALL TO SUBROUTINE emint. THIS HAPPENS WHEN
C  IHESS = -1, WHICH MEANS THAT THE HESSIAN MATRIX IN THE INTERIOR
C  POINT EM ALGORITHM WAS SINGULAR. RATHER THAN SIMPLY STOPPING AS IT
C  DID PREVIOUSLY, NOW THE PROGRAM WILL CREATE THE OUTPUT FILES BEFORE
C  STOPPING ... BASED ON THE VALUES FROM THE PREVIOUS CYCLE.
C  FIRST, WRITE THE REASON FOR STOPPING AS ICONVERGE = 3 BELOW. THEN
C  RESET CORDEN BACK TO CORDLAST (SEE ABOVE), WHICH WAS THE CORDEN
C  AT THE END OF THE PREVIOUS CYCLE.
C  FOR NPAGFULL, OF COURSE, NO WRITING OCCURS TO OUTPUT FILES.


C  WRITE WHY THE PROGRAM STOPPED.

        IF(IHESS .EQ. -1) THEN

         NACTVE = NACTLAST

         DO I = 1,NACTVE
          DO J = 1,NVAR+1
           CORDEN(I,J) = CORDLAST(I,J)
          END DO
         END DO

         GO TO 910

        ENDIF


C  Starting with bigmlt1.f, this is an entry point to continue 
c  calculations


  910 CONTINUE

cbegin endgame
c we can only arrive here from the control section, which menas
c that we ahve completed optimizaiton but not done the subsequent
c expansion.  This means that the density is correct, and we can safely
c just write it out and exit.

      WRITE(*,1294) ICYCLE,MAXCYC
 1294 FORMAT(/' NPAG RAN ',I6,' OUT OF A MAXIMUM POSSIBLE ',I6/
     1' CYCLES TO OBTAIN THE POSTERIOR DENSITY.')

C  FOR NPAGFULL, THE DENSITY IS CORRECT AT THIS POINT. SO RETURN TO
C  THE BESTDOSxxx PROGRAM.

C  AS OF NPAGFULLA11.FOR, CLOSE FILE 37.

      CLOSE(37)




	RETURN
	END


