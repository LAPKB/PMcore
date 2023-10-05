#PRI

V1,5,160                
CL1,4,9              
V2,100,200                
CL2,25,35              
POPMAX,100000000,100000000000            
Kgs,0.01,0.25            
Kks,0.01,0.5              
E50_1s,0.1,2.5            
E50_2s,0.1,10            
alpha_s,-8,5         
Kgr1,0.004,0.1            
Kkr1,0.08,0.4           
E50_1r1,8,17
alpha_r1,-8,5       
Kgr2,0.004,0.3           
Kkr2,0.1,0.5           
E50_2r2,5,8         
alpha_r2,-5,5
INIT_3,0
INIT_4,-1,4
INIT_5,-1,3
H1s,0.5,8            
H2s,0.1,4              
H1r1,5,25                      
H2r2,10,22

C Drug 1 is linezolid, Drug 2 is rifampin

#COV

IC_T
IC_R1
IC_R2

#SEC
E50_2r1 = E50_2s         
E50_1r2 = E50_1s 
H2r1 = H2s            
H1r2 = H1s     
XM0BEST=0
XM0Converge=0


#OUT

Y(1) = X(1)/V1           
Y(2) = X(2)/V2           
Y(3) = DLOG10(XNs + XNr1 + XNr2)             
Y(4) = DLOG10(XNr1)      
Y(5) = DLOG10(XNr2)  


#INI

X(3) = 10.D0**IC_T      
X(4) = 10.D0**INIT_4 
X(5) = 10.D0**INIT_5

#ERR

G=1
0.5000000000E-01,  0.8000000000E-01,   0.000000000,      0.000000000    
0.5000000000E-01,  0.8000000000E-01,   0.000000000,       0.000000000    
0.1000000000,      0.1000000000,       0.000000000,       0.000000000    
0.1000000000,     0.1000000000,       0.000000000,       0.000000000    
0.1000000000,      0.1000000000,       0.000000000,       0.000000000    
       


#DIFF

COMMON/TOBESTM0/U,V,W,H1,H2,XX                    
EXTERNAL BESTM0   
DIMENSION START(7),STEP(7)  

C  FOR DRUTRU13.FOR, X(1) AND X(2) ARE SET = 0 IF THEY ARE PASSED TO        
C  THIS ROUTINE AS NEGATIVE (SINCE THAT IS A NUMERICAL ARTIFACT FOR         
C  THIS MODEL).           

[format]                          
      IF(X(1) .LT. 0.D0) X(1) = 0.D0               
      IF(X(2) .LT. 0.D0) X(2) = 0.D0               
                          
                          
	XP(1) = RATEIV(1) - CL1*X(1)/V1                   
	XP(2) = RATEIV(2) - CL2*X(2)/V2                   
                          
                          
C  CALCULATE PIECES FOR X(3) = XNs, X(4) = XNr1, AND X(5) = XNr2.           
                          
	XNs = X(3)               
	XNr1 = X(4)              
	XNr2 = X(5) 
             
      E = 1.D0 - (XNs + XNr1 + XNr2)/POPMAX             
             
             
C  CALCULATE VALUES NEEDED FOR THE XP(3) EQ.
                    
	D1 = X(1)/V1             
	D2 = X(2)/V2             
	U = D1/E50_1s            
	V = D2/E50_2s            
	W = alpha_s*D1*D2/E50_1s/E50_2s                   
	H1 = 1.D0/H1s            
	H2 = 1.D0/H2s            
	XX = (H1 + H2)/2.D0      
                          
C  CALCULATE XM0BEST. IF U = 0, IT IS A SIMPLE FUNCTION OF V; IF V = 0,        
C  IT IS A SIMPLE FUNCTION OF U. OTHERWISE, WILL HAVE TO CALL ELDERY       
C  (THE NELDER MEED ALGORITHM) TO FIND IT.

C  BUT FIRST TEST TO SEE IF BOTH U AND V ARE VERY SMALL. IF SO, THE 
C  GRECO EQ. IS ESSENTIALLY SAYING THAT XM0BEST SHOULD BE 0. WE WILL
C  ARBIRTRARILY SET THE THRESHOLD FOR "SMALL" TO BE 1.D-5.

      IF(U .LE. 1.D-5 .AND. V .LE. 1.D-5) THEN
       XM0BEST = 0.D0
C       GO TO 110
C  110	XMs = XM0BEST/(1.D0 + XM0BEST)                
  	XMs = XM0BEST/(1.D0 + XM0BEST)                
	XP(3) = XNs*(Kgs*E - Kks*XMs)
C From here, skip to the XP(4) calculation,
C
      ELSE

C this endif is now moved to after the XP(3) calculation
C     ENDIF
                          
       IF(V .LE. 0.D0) XM0BEST = U**(1.D0/H1)       
       IF(U .LE. 0.D0) XM0BEST = V**(1.D0/H2)
C
C This condition searches for monotherapy.  If only D1 or D2 is applied,
C then ONLY U or V can be 0 at this point. And if one of these is 0,
C then we do not have to optimize for XM0BEST, we can calculate it.
C The condition in the above block states that "for combination therapy,
C if both drugs are at a small concentration, there is no synergy. Thus
C XM0BEST = 0"
C
C Now, we calculate XM0BEST for the case where both D1 and D2 are at
C interactable concentrations.
C
       IF(V .GT. 0.D0 .AND. U .GT. 0.D0) THEN         
                          
        START(1) = .00001  

C  NOTE THAT THE STARTING VALUES IN START(1) FOR ELDERY ARE ALL 
C  FIXED = .00001, SINCE I DISCOVERED IN ELDM.FOR THAT THOUGH THE
C  NELDER MEED ALGORITHM CAN BE VERY UNSTABLE IN CERTAIN CONDITIONS, A
C  STARTING VALUE OF .00001 USUALLY LEADS IMMEDIATELY TO THE CORRECT
C  VALUE OF M0 TO SOLVE THE GRECO EQ. (AT LEAST FOR THE EXAMPLES I DID
C  ON PAGES 12 - 15 OF 12/10/12 NOTES). 
                                                        
C  SET TOL = THE TOLERANCE DESIRED FOR THE MINIMIZATION ROUTINE.            
                          
        TOL = 1.D-10                                      
        STEP(1)= -.2D0*START(1) 
                                            
        CALL ELDERY(1,START,XM0BEST1,VALMIN1,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)                        
                          
C  XM0BEST1 = THE VALUE OF XM0 WHICH GIVES THE MINIMIZED SQUARED             
C            DIFFERENCE IN SUBROUTINE BESTM0.      
                          
C  VALMIN1 = MIN. VALUE OF FUNCTION ACHIEVED, THIS VALUE NOT CURRENTLY       
C  USED.                  
                          
C  ICONV = 1 IF ELDERY CONVERGED; 0 OTHERWISE.     
                          
	IF(ICONV .EQ. 0) THEN    

	 WRITE(*,9021)           
 9021 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR s.'/)

         WRITE(*,*)' FOR THE XP(3) EQ.... '
         WRITE(*,123) XM0BEST1,VALMIN1 
  123    FORMAT(/' THE EST. FOR M0 FROM ELDERY WAS  ',G20.12/
     3' AND THIS GAVE A VALMIN OF ',G20.12//)
        WRITE(*,129) D1,D2,U,V,W,ALPHA_S,H1,H2
  129   FORMAT(//' NOTE THAT D1,D2 = ',2(G20.12,2X)/
     1' U,V = ',2(G20.12,2X)/
     2' W,ALPHA_S = ',2(G20.12,2X)/
     3' H1,H2 = ',2(G20.12,2X))

            
	 PAUSE                   
	 STOP                    
	ENDIF  

C  IF VALMIN1 .LE. 1.D-10, XM0BEST1 IS A GOOD MATCH TO THE GRECO EQ.,
C  GIVEN THE INPUT PARAMETER VALUES. IF VALMIN1 .GT. 1.D-10, XM0BEST1
C  MAY NOT BE A GOOD MATCH. IN THIS CASE, CALL FINDM0 (BASED ON
C  FINDM03) TO OBTAIN A GOOD ESTIMATE FOR XM0BEST WHICH WILL THEN BE
C  USED AS THE INITIAL ESTIMATE FOR ANOTHER CALLING OF ELDERY. THEN USE
C  THE XM0BEST WHICH HAS THE SMALLER VALMIN FROM THE TWO ELDERY CALLS.
 


        IF(VALMIN1 .LE. 1.D-10) XM0BEST = XM0BEST1


        IF(VALMIN1 .GT. 1.D-10) THEN

         CALL FINDM0(U,V,alpha_s,H1,H2,XM0EST)

C  NOTE THAT IF XM0EST RETURNS AS -1, IT MEANS THAT FINDM0 COULD NOT
C  SOLVE THE GRECO EQ. (IT IS UNSOLVABLE), AND IN THIS CASE, JUST USE
C  THE XM0BEST1 FROM ELDERY, REGARDLESS OF VALMIN1, SINCE THAT IS THE
C  BEST VALUE FOR M0 THAT CAN BE OBTAINED.

         IF(XM0EST .LT. 0.D0) THEN
          XM0BEST = XM0BEST1
C          GO TO 110
C         ENDIF
         ELSE

          START(1) = XM0EST
          STEP(1)= -.2D0*START(1) 
          CALL ELDERY(1,START,XM0BEST2,VALMIN2,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT) 

C  NOW USE WHICHEVER ESTIMATE FROM ELDERY HAD THE SMALLER VALMIN.

          XM0BEST = XM0BEST1
          IF(VALMIN2 .LT. VALMIN1) XM0BEST = XM0BEST2                       

C  ICONV = 1 IF ELDERY CONVERGED; 0 OTHERWISE.     
                          
	  IF(ICONV .EQ. 0) THEN
	   WRITE(*,8021)           
 8021 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR s.'//
     1' EVEN AFTER FINDM0 WAS USED. '/)            
	   PAUSE                   
	   STOP  
	  ENDIF 

         ENDIF
C  THE ABOVE ENDIF IS FOR THE  IF(XM0EST .LT. 0.D0)  CONDITION.

        ENDIF
C  THE ABOVE ENDIF IS FOR THE  IF(VALMIN1 .GT. 1.D-10  CONDITION.

       ENDIF
C  THE ABOVE ENDIF IS FOR THE  IF(V .GT. 0.D0 .AND. U .GT. 0.D0) 
C  CONDITION.        

  110	XMs = XM0BEST/(1.D0 + XM0BEST)                
	XP(3) = XNs*(Kgs*E - Kks*XMs)                   

C This endif is for the U && V less than 10^-5 condition       
     ENDIF

      
C  CALCULATE VALUES NEEDED FOR THE XP(4) EQ.                   
                                                    
	D1 = X(1)/V1             
	D2 = X(2)/V2             
	U = D1/E50_1r1                                     
	V = D2/E50_2r1           
	W = alpha_r1*D1*D2/E50_1r1/E50_2r1                
	H1 = 1.D0/H1r1           
	H2 = 1.D0/H2r1           
	XX = (H1 + H2)/2.D0 

      IF(U .LE. 1.D-5 .AND. V .LE. 1.D-5) THEN

C       XM0BEST = 0.D0
C       GO TO 210
C      ENDIF
C
C  210	XMr1 = XM0BEST/(1.D0 + XM0BEST)                   
C	XP(4) = XNr1*(Kgr1*E - Kkr1*XMr1)               
C
C Above can be replaced with:
       XP(4) = XNr1*Kgr1*E

      ELSE

                          
       IF(V .LE. 0.D0) XM0BEST = U**(1.D0/H1)       
       IF(U .LE. 0.D0) XM0BEST = V**(1.D0/H2)

                          
       IF(V .GT. 0.D0 .AND. U .GT. 0.D0) THEN         
                          
        START(1) = .00001  
        TOL = 1.D-10                                      
        STEP(1)= -.2D0*START(1) 
                                            
        CALL ELDERY(1,START,XM0BEST1,VALMIN1,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)                        
                          
        IF(ICONV .EQ. 0) THEN    
	 WRITE(*,9022)           
 9022 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR r1.'/)            
	 PAUSE                   
	 STOP                    
        ENDIF  

        IF(VALMIN1 .LE. 1.D-10) XM0BEST = XM0BEST1

        ELSE
C       IF(VALMIN1 .GT. 1.D-10) THEN

         CALL FINDM0(U,V,alpha_r1,H1,H2,XM0EST)

         IF(XM0EST .LT. 0.D0) THEN
          XM0BEST = XM0BEST1
C          GO TO 210
C         ENDIF
         ELSE

          START(1) = XM0EST
          STEP(1)= -.2D0*START(1) 
          CALL ELDERY(1,START,XM0BEST2,VALMIN2,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT) 
          XM0BEST = XM0BEST1
          IF(VALMIN2 .LT. VALMIN1) XM0BEST = XM0BEST2                       
	  IF(ICONV .EQ. 0) THEN    
	   WRITE(*,8022)           
 8022 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR r1.'//
     1' EVEN AFTER FINDM0 WAS USED. '/)            
	   PAUSE                   
	   STOP                    
	  ENDIF
         ENDIF
C The ABOVE END IF is for the IF(XM0EST .LT. 0.D0) THEN condition

        ENDIF
C  THE ABOVE ENDIF IS FOR THE  IF(VALMIN1 .GT. 1.D-10  CONDITION.
 
       ENDIF
C  THE ABOVE ENDIF IS FOR THE  IF(V .GT. 0.D0 .AND. U .GT. 0.D0) CONDITION.

  210  XMr1 = XM0BEST/(1.D0 + XM0BEST)                   
       XP(4) = XNr1*(Kgr1*E - Kkr1*XMr1)               

      ENDIF
C  ABOVE ENDIF is for the (U AND V .LE 1.D-5) condition


C  CALCULATE VALUES NEEDED FOR THE XP(5) EQ.
                          
	D1 = X(1)/V1             
	D2 = X(2)/V2             
	U = D1/E50_1r2           
	V = D2/E50_2r2           
	W = alpha_r2*D1*D2/E50_1r2/E50_2r2                
	H1 = 1.D0/H1r2           
	H2 = 1.D0/H2r2           
	XX = (H1 + H2)/2.D0 

      IF(U .LE. 1.D-5 .AND. V .LE. 1.D-5) THEN

C       XM0BEST = 0.D0
C       GO TO 310
C  310	XMr2 = XM0BEST/(1.D0 + XM0BEST)                   

       XMr2 = XM0BEST/(1.D0 + XM0BEST)                   
       XP(5) = XNr2*(Kgr2*E - Kkr2*XMr2) 

      ELSE

       IF(V .LE. 0.D0) XM0BEST = U**(1.D0/H1)       
       IF(U .LE. 0.D0) XM0BEST = V**(1.D0/H2)

       IF(V .GT. 0.D0 .AND. U .GT. 0.D0) THEN         
                          
	START(1) = .00001  
        TOL = 1.D-10                                      
        STEP(1)= -.2D0*START(1) 
        CALL ELDERY(1,START,XM0BEST1,VALMIN1,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)                        
                          
	IF(ICONV .EQ. 0) THEN    
	 WRITE(*,9023)           
 9023 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR r2.'/)            
	 PAUSE                   
	 STOP                    
	ENDIF  
 
        IF(VALMIN1 .LE. 1.D-10) XM0BEST = XM0BEST1


C  IF(VALMIN1 .GT. 1.D-10) THEN
        ELSE

         CALL FINDM0(U,V,alpha_r2,H1,H2,XM0EST)

         IF(XM0EST .LT. 0.D0) THEN
          XM0BEST = XM0BEST1

C          GO TO 310
C  310  XMr2 = XM0BEST/(1.D0 + XM0BEST)                   
C       XP(5) = XNr2*(Kgr2*E - Kkr2*XMr2) 

         ELSE

          START(1) = XM0EST
          STEP(1)= -.2D0*START(1) 
          CALL ELDERY(1,START,XM0BEST2,VALMIN2,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT) 

          XM0BEST = XM0BEST1

          IF(VALMIN2 .LT. VALMIN1) XM0BEST = XM0BEST2                       
                          
	  IF(ICONV .EQ. 0) THEN    
	   WRITE(*,8023)           
 8023 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR r2.'//
     1' EVEN AFTER FINDM0 WAS USED. '/)            
	   PAUSE                   
	   STOP                    
	  ENDIF 

         ENDIF

        ENDIF
C  THE ABOVE ENDIF IS FOR THE  IF(VALMIN1 .GT. 1.D-10  CONDITION.

       ENDIF
C  THE ABOVE ENDIF ENDS IF(V .GT. 0.D0 .AND. U .GT. 0.D0) CONDITION.

  310  XMr2 = XM0BEST/(1.D0 + XM0BEST)                   
       XP(5) = XNr2*(Kgr2*E - Kkr2*XMr2) 

      ENDIF
C The ABOVE END IF(U .LE. 1.D-5 .AND. V .LE. 1.D-5) THEN condition
[/format]

#OUT
Y(1) = X(1)/V1           
Y(2) = X(2)/V2           
Y(3) = DLOG10(XNs + XNr1 + XNr2)             
Y(4) = DLOG10(XNr1)      
Y(5) = DLOG10(XNr2)  


#Init
X(3) = 10.D0**IC_T      
X(4) = 10.D0**INIT_4 
X(5) = 10.D0**INIT_5

#ERR
G=1
0.5000000000E-01,  0.8000000000E-01,   0.000000000,      0.000000000    
0.5000000000E-01,  0.8000000000E-01,   0.000000000,       0.000000000    
0.1000000000,      0.1000000000,       0.000000000,       0.000000000    
0.1000000000,     0.1000000000,       0.000000000,       0.000000000    
0.1000000000,      0.1000000000,       0.000000000,       0.000000000    


#EXTRA
        SUBROUTINE ELDERY(N,START,XMIN,YNEWLO,REQMIN,STEP,ITMAX,FUNCC,IPRINT,ICONV,NITER,ICOUNT)     
                          
C  ELDERY DIFFERS FROM ELDERX ONLY IN THE DIMENSION STATEMENT. ALL 5'S      
C  ARE CHANGED TO 7'S, AND ALL 6'S ARE CHANGED TO 8'S. THIS ALLOWS 7        
C  PARAMETERS INSTEAD OF JUST 5.                   

                          
C  ELDERX DIFFERS FROM ELDER (DESCRIBED BELOW) ONLY IN THAT N, THE          
C  DIMENSION OF START (THE NO. OF UNKNOWN PARAMETERS OVER WHICH THE         
C  MINIMIZATION IS DONE) IS PASSED TO THE SUBROUTINE FUNCC IN THE CALLING   
C  STATEMENTS.            
C                         
C  ELDER IS A PROGRAM TO MINIMIZE A FUNCTION USING THE NELDER-MEED          
C  ALGORITM.              
C    THE CODE WAS ADAPTED FROM A PROG. IN J. OF QUALITY TECHNOLOGY VOL.     
                          
C    JAN. 1974. BY D.M. OLSSON.                    
C  CALLING ARGUMENTS:     
C    N     -NUMBER OF UNKNOWN PARAMS. UP TO 99.    
C    START -A VECTOR WITH THE INITIAL QUESSES OF THE SOLUTION PARAMS.       
C    ITMAX -THE MAXIMUM NUMBER OF ITERATIONS.      
C             (KCOUNT IS THE MAX NUM OF FUNCC CALLS.SET AT 1000000)         
C    STEP  -THE STEP SIZE VECTOR FOR DEFINING THE N ADDITIONAL              
C             VERTICIES.  
C    REQMIN-THE STOP TOLERANCE.                    
C    XMIN   -THE SOLUTION VECTOR.                  
C    YNEWLO-THE FUCTION VALUE AT XMIN.             
C    IPRINT-SWITCH WHICH DETERMINES IF INTERMEDIATE ITERATIONS              
C              ARE TO BE PRINTED. (0=NO,1=YES).    
C    ICONV -FLAG INDICATING WHETHER OR NOT CONVERGENCE HAS BEEN             
C             ACHEIVED. (0=NO,1=YES).              
C    NITER -THE NUMBER OF ITERATIONS PERFORMED.    
C    ICOUNT-THE NUMBER OF FUNCTION EVALUATIONS.    
C    FUNCC  -THE NAME OF THE SUBROUTINE DEFINING THE FUNCTION.              
C             THIS SUBROUTINE MUST EVALUATE THE FUNCTION GIVEN A            
C             VALUE FOR THE PARAMETER VECTOR. THE ROUTINE IS OF             
C             THE FOLLOWING FORM:                  
C               FUNCC(P,FV), WHERE P IS THE PARAMETER VECTOR,               
C                             AND FV IS THE FUNCTION VALUE.                 
C  A SUBROUTINE TO PRINT THE RESULTS OF ITERMEDIATE ITERATIONS              
C    MUST ALSO BE SUPPLIED. ITS NAME AND CALLING SEQUENCE ARE               
C    DEFINED AS FOLLOWS:  
C      PRNOUT(P,N,NITER,NFCALL,FV).                
C  OTHER PROGRAM VARIABLES OF INTEREST ARE;        
C    XSEC  -THE COORDINATES OF THE VETEX WITH THE 2ND SMALLEST FUNCTION     

C             VALUE.      
C    YSEC  - THE FUNCTION VALUE AT XSEC.           
C                         
      IMPLICIT REAL*8(A-H,O-Z)                     
        DIMENSION START(N),STEP(N),XMIN(N),XSEC(7),P(7,8),PSTAR(7),P2STAR(7),PBAR(7),Y(8)     
        EXTERNAL FUNCC    
        DATA RCOEFF/1.0D0/,ECOEFF/2.0D0/,CCOEFF/.5D0/                       
        KCOUNT=1000000    
        ICOUNT=0          
        NITER=0           
        ICONV=0           
C                         
C  CHECK INPUT DATA AND RETURN IF AN ERROR IS FOUND.                        
C                         
        IF(REQMIN.LE.0.0D0) ICOUNT=ICOUNT-1        
        IF(N.LE.0) ICOUNT=ICOUNT-10                
        IF(N.GT.99) ICOUNT=ICOUNT-10               
        IF(ICOUNT.LT.0) RETURN                     
C                         
C  SET INITIAL CONSTANTS  
C                         
        DABIT=2.04607D-35 
        BIGNUM=1.0D+38    
        KONVGE=5          
        XN=FLOAT(N)       
        DN=FLOAT(N)       
        NN=N+1    
[format]        
C                         
C  CONSTRUCTION OF INITIAL SIMPLEX.                
C                         
1001    DO 1 I=1,N        
1       P(I,NN)=START(I)  
        CALL FUNCC(N,START,FN)                     
        Y(NN)=FN          
        ICOUNT=ICOUNT+1   
C       CALL PRNOUT(START,N,NITER,ICOUNT,FN)       
        IF(ITMAX.NE.0) GO TO 40                    
        DO 45 I=1,N       
45      XMIN(I)=START(I)  
        YNEWLO=FN         
        RETURN            
40      DO 2 J=1,N        
        DCHK=START(J)     
        START(J)=DCHK+STEP(J)                      
        DO 3 I=1,N        
3       P(I,J)=START(I)   
        CALL FUNCC(N,START,FN)                     
        Y(J)=FN           
        ICOUNT=ICOUNT+1   
2       START(J)=DCHK     
C                         
C  SIMPLEX CONSTRUCTION COMPLETE.                  
C                         
C    FIND THE HIGHEST AND LOWEST VALUES. YNEWLO (Y(IHI)) INDICATES THE      
C     VERTEX OF THE SIMPLEX TO BE REPLACED.        
C                         
1000    YLO=Y(1)          
        YNEWLO=YLO        
        ILO=1             
        IHI=1             
        DO 5 I=2,NN       
        IF(Y(I).GE.YLO) GO TO 4                    
        YLO=Y(I)          
        ILO=I             
4       IF(Y(I).LE.YNEWLO) GO TO 5                 
        YNEWLO=Y(I)       
        IHI=I             
5       CONTINUE          
C                         

        IF(ICOUNT.LE.NN) YOLDLO=YLO                
        IF(ICOUNT.LE.NN) GO TO 2002                
        IF(YLO.GE.YOLDLO) GO TO 2002               
        YOLDLO=YLO        
        NITER=NITER+1     
        IF(NITER.GE.ITMAX) GO TO 900               
        IF(IPRINT.EQ.0) GO TO 2002                 
C       CALL PRNOUT(P(1,ILO),N,NITER,ICOUNT,YLO)   
C                         
C  PERFORM CONVERGENCE CHECKS ON FUNCTIONS.        
C                         
2002    DCHK=(YNEWLO+DABIT)/(YLO+DABIT)-1.0D0      
        IF(DABS(DCHK).GT. REQMIN) GO TO 2001       
        ICONV=1           
        GO TO 900         
C                         
2001    KONVGE=KONVGE-1   
        IF(KONVGE.NE.0) GO TO 2020                 
        KONVGE=5          
C                         
C  CHECK CONVERGENCE OF COORDINATES ONLY EVERY 5 SIMPLEXES.                 
C                         
        DO 2015 I=1,N     
        COORD1=P(I,1)     
        COORD2=COORD1     
        DO 2010 J=2,NN    
        IF(P(I,J).GE.COORD1) GO TO 2005            
        COORD1=P(I,J)     
2005    IF(P(I,J).LE.COORD2) GO TO 2010            
        COORD2=P(I,J)     
2010    CONTINUE          
        DCHK=(COORD2+DABIT)/(COORD1+DABIT)-1.0D0   
        IF(DABS(DCHK).GT.REQMIN) GO TO 2020        
2015    CONTINUE          
        ICONV=1           
        GO TO 900         
2020    IF(ICOUNT.GE.KCOUNT) GO TO 900             
C                         
C  CALCULATE PBAR, THE CENTRIOD OF THE SIMPLEX VERTICES EXCEPTING THAT      
C   WITH Y VALUE YNEWLO.  
C                         
        DO 7 I=1,N        
        Z=0.0D0           
        DO 6 J=1,NN       
6       Z=Z+P(I,J)        
        Z=Z-P(I,IHI)      
7       PBAR(I)=Z/DN      
C                         
C  REFLECTION THROUGH THE CENTROID.                
C                         
        DO 8 I=1,N        
8       PSTAR(I)=(1.0D0+RCOEFF)*PBAR(I)-RCOEFF*P(I,IHI)                     
        CALL FUNCC(N,PSTAR,FN)                     
        YSTAR=FN          
        ICOUNT=ICOUNT+1   
        IF(YSTAR.GE.YLO) GO TO 12                  
        IF(ICOUNT.GE.KCOUNT) GO TO 19              
C                         
C  SUCESSFUL REFLECTION SO EXTENSION.              
C                         
        DO 9 I=1,N        
9       P2STAR(I)=ECOEFF*PSTAR(I)+(1.0D0-ECOEFF)*PBAR(I)                    
        CALL FUNCC(N,P2STAR,FN)                    
        Y2STAR=FN         
        ICOUNT=ICOUNT+1   
C                         
C  RETAIN EXTENSION OR CONTRACTION.                
C                         
        IF(Y2STAR.GE.YSTAR) GO TO 19               
10      DO 11 I=1,N       
11      P(I,IHI)=P2STAR(I)
        Y(IHI)=Y2STAR     
        GO TO 1000        
C                         
C  NO EXTENSION.          
C                         
12      L=0               
        DO 13 I=1,NN      
        IF(Y(I).GT.YSTAR) L=L+1                    
13      CONTINUE          
        IF(L.GT.1) GO TO 19                        
        IF(L.EQ.0) GO TO 15                        
C                         
C  CONTRACTION ON REFLECTION SIDE OF CENTROID.     
C                         
        DO 14 I=1,N       
14      P(I,IHI)=PSTAR(I) 

        Y(IHI)=YSTAR      
C                         
C  CONTRACTION ON THE Y(IHI) SIDE OF THE CENTROID. 
C                         
15      IF(ICOUNT.GE.KCOUNT) GO TO 900             
        DO 16 I=1,N       
16      P2STAR(I)=CCOEFF*P(I,IHI)+(1.0D0-CCOEFF)*PBAR(I)                    
        CALL FUNCC(N,P2STAR,FN)                    
        Y2STAR=FN         
        ICOUNT=ICOUNT+1   
        IF(Y2STAR.LT.Y(IHI)) GO TO 10              
C                         
C  CONTRACT THE WHOLE SIMPLEX                      
C                         
        DO 18 J=1,NN      
        DO 17 I=1,N       
        P(I,J)=(P(I,J)+P(I,ILO))*0.5D0             
17      XMIN(I)=P(I,J)    
        CALL FUNCC(N,XMIN,FN)                      
        Y(J)=FN           

18      CONTINUE          
        ICOUNT=ICOUNT+NN  
        IF(ICOUNT.LT.KCOUNT) GO TO 1000            
        GO TO 900         
C                         
C  RETAIN REFLECTION.     
C                         
                          
19      CONTINUE          
        DO 20 I=1,N       
20      P(I,IHI)=PSTAR(I) 
        Y(IHI)=YSTAR      
        GO TO 1000        
C                         
C  SELECT THE TWO BEST FUNCTION VALUES (YNEWLO AND YSEC) AND THEIR          
C    COORDINATES (XMIN AND XSEC)>                  
C                         
900     DO 23 J=1,NN      
        DO 22 I=1,N       
22      XMIN(I)=P(I,J)    
        CALL FUNCC(N,XMIN,FN)                      
        Y(J)=FN           
23      CONTINUE          
        ICOUNT=ICOUNT+NN  
        YNEWLO=BIGNUM     
        DO 24 J=1,NN      
        IF(Y(J).GE.YNEWLO) GO TO 24                
        YNEWLO=Y(J)       
        IBEST=J           
24      CONTINUE          
        Y(IBEST)=BIGNUM   
        YSEC=BIGNUM       
        DO 25 J=1,NN      
        IF(Y(J).GE.YSEC) GO TO 25                  
        YSEC=Y(J)         
        ISEC=J            
25      CONTINUE          
        DO 26 I=1,N       
        XMIN(I)=P(I,IBEST)
        XSEC(I)=P(I,ISEC) 
26      CONTINUE          
        RETURN            
        END               
	                         
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC              
C                                                   
	SUBROUTINE BESTM0(NC,VEC,FNTVAL)                  
	IMPLICIT REAL*8(A-H,O-Z) 
	DIMENSION VEC(NC)        
	COMMON/TOBESTM0/U,V,W,H1,H2,XX                    
                          
C  COMMON/TOBESM0 IS SUPPLIED FROM MAIN.           
                          
C  THIS ROUTINE, CALLED BY ELDERY, FINDS THE FUNCTIONAL VALUE, FNTVAL       
C  FOR THE SUPPLIED VARIABLE VECTOR, VEC.          
                          
	XM0 = VEC(1)             
	                         
	T1 = U/XM0**H1           
	T2 = V/XM0**H2           
	T3 = W/XM0**XX           
	FNTVAL = (1.D0 - T1 - T2 - T3)**2.D0              
                          
	RETURN                   
	END 
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC              
C                                                   
      SUBROUTINE FINDM0(UFINAL,V,ALPHA,H1,H2,XM0EST)
	IMPLICIT REAL*8(A-H,O-Z) 

C  THIS ROUTINE, CALLED BY DIFFEQ, INPUTS UFINAL,V,ALPHA,H1, AND H2, AND
C  THEN CALCULATES THE BEST M0 VALUE XM0EST) TO SATISFY THE GRECO EQ. 
C  THE LOGIC OF THIS ROUTINE IS BASED ON FINDM03.FOR, WITH NOINT = 1000.

C  CALCULATE THE APPROXIMATE VALUE OF M0 USING THE FOLLOWING 1ST
C  ORDER APPROXIMATION, NOINT TIMES:
C  M0(U + DELU) ~ M0(U) + M0'(U)*DELU, UNTIL U + DELU = UFINAL.
C  NOTE THAT M0'(U) IS GIVEN BY EQ. 6 ON PG. 8 OF 12/10/12 NOTES.
C  AND NOTE THAT DELU = UFINAL/NOINT.

      NOINT = 1000
      DELU = UFINAL/NOINT

C  M0(U = 0) = V**(1/H2) FROM EQ. 0 ON PG. 1 OF 12/10/12 NOTES.

      XM = V**(1.D0/H2) 
      U = 0.D0

C  U IS THE CURRENT VALUE OF U.
C  XM = WILL BE THE RUNNING VALUE OF M0(U). 

      HH = (H1 + H2)/2.D0


      DO INT = 1,NOINT  
   
C  XM IS THE CURRENT VALUE OF M0(U). FIND M0'(U) USING EQ. 6 ON PG.
C  8 OF 12/10/12 NOTES. CALL IT XMP.

       TOP = 1.D0/XM**H1 + ALPHA*V/XM**HH
       B1 = U*H1/XM**(H1 + 1.D0)
       B2 = V*H2/XM**(H2 + 1.D0)
       B3 = ALPHA*V*U*HH/XM**(HH + 1.D0)
       XMP = TOP/(B1 + B2 + B3)

C  NOW APPLY THE ABOVE EQ. TO GET M0(U + DELU), WHICH WILL BE THE
C  UPDATE TO XM. AND INCREASE U TO ITS CURRENT VALUE.

       XM = XM + XMP*DELU

C  NOTE THAT IF XM EVER BECOMES .LE. 0, IT MEANS THAT THE GRECO
C  EQ. IS NOT SOLVABLE (SEE REAL3.EXP COMMENTS IN REMARK 13.b). AND
C  IN THIS CASE, SET XM0EST = -1, AND RETURN.

       IF(XM .LE. 0.D0) THEN
        XM0EST = -1.D0
        RETURN
       ENDIF


       U = DELU*INT

      END DO

C  THE ABOVE END DO IS FOR THE  DO INT = 1,NOINT  LOOP.
            
      XM0EST = XM

      RETURN
      END
[/format]