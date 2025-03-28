C  MODEL3.FOR                                              8/15/13

C  MODEL3 IS THE SAME AS MODEL2, EXCEPT THAT IT HAS AN ADDITIONAL
C  OUTPUT EQ. IN SUBROUTINE OUTPUT (Y(2) = X(3)).

C-----------------------------------------------------------------------

C  MODEL2.FOR                                              8/13/13

C  MODEL2 ALSO HAS N = -1, BUT INSTEAD OF A 2-COMP. MODEL AS IS 
C  MODEL1.FOR, IT WILL BE A STANDARD 3-COMP. MODEL, WITH PARAMETERS
C  KEL, VOL, KA, KCP, KPC.

C-----------------------------------------------------------------------

C  MODEL1.FOR                                              5/16/13

C  MODEL1 HAS N = -1 IN SUBROUTINE SYMBOL; THIS WILL BE A 2-COMPARTMENT
C  LINEAR MODEL, WITH PARAMETERS KEL, VOL, KA.

C-----------------------------------------------------------------------

C  TSTMULTK.FOR                                            OCT, 2012

C  TSTMULTK HAS THE FOLLOWING CHANGES FROM TSTMULTJ:

C  THE TWO DECLARATION STATEMENTS SHOWN IN TSTMULTJ.FOR COMMENTS 
C  WILL NOW BE ELIMINATED IN ALL BUT SUBROUTINES ANAL3, AND CASE1,
C  CASE2, CASE3, AND CASE4. THE REASON IS THAT, WHEREAS THE P(.)
C  AND CV(.) ARRAYS ARE ALWAYS SET JUST BEFORE ANY OF THE ROUTINES OF
C  THIS MODEL FILE ARE CALLED, THE SAME CANNOT BE SAID FOR THE 
C  PARAMETERS IN COMMON/RATESV. THOSE VALUES ARE ONLY SET IN 
C  SUBROUTINE ANAL3, WHICH MAY NOT HAVE BEEN CALLED JUST BEFORE ONE
C  OF THE OTHER ROUTINES OF THIS FILE IS CALLED. THAT WOULD MEAN THAT
C  THOSE PARAMETERS MIGHT NOT HAVE UP-TO-DATE VALUES IN THEM. TO
C  AVOID THIS, THE USER WILL NOW HAVE TO CODE HIS ROUTINES USING
C  P(.) AND CV(.), WHICH, AS INDICATED ABOVE, WILL BE UP TO DATE.

C-----------------------------------------------------------------------

C  TSTMULTJ.FOR                                            SEPT, 2012

C  TSTMULTJ HAS THE FOLLOWING CHANGES TO TSTMULTI:

C  THE FOLLOWING TWO DECLARATION STATEMENTS ...
C        DOUBLE PRECISION KE,KA,KCP,KPC     AND
C        COMMON /RATESV/ KE,KA,KCP,KPC,V       
C  WHICH WERE ONLY IN SUBROUTINES OUTPUT AND ANAL3, WILL NOW ALSO BE IN
C  SUBROUTINES DIFFEQ, GETFA, GETIX, AND GETTLAG. THIS WILL ALLOW THE
C  USER TO REFER TO THESE PARAMETERS BY NAME, RATHER THAN BY P(.), IN
C  ALL THE ROUTINES OF THE MODEL.

C-----------------------------------------------------------------------

C  TSTMULTI.FOR                                            APR, 2011

C  TSTMULTI HAS THE FOLLOWING CHANGES TO TSTMULTH:

C  THE CODE TO ESTABLISH THE COVARIATE VALUES IN THE PATIENT DATA
C  FILES IS CHANGED. NOW, INSTEAD OF 2 + NADD COVARIATES, THERE WILL BE
C  ONLY NADD. THIS REQUIRES ALL THE DO LOOPS FROM 1 TO 2 + NADD TO NOW
C  BE FROM 1 TO NADD. 

C  IN ADDITION, THE WARNINGS ABOUT THE DIFFERENCES BETWEEN THE 
C  COVARIATE VALUES (SPECIFICALLY THOSE INVOLVING WT AND CCR) 
C  DEPENDING ON WHETHER THE USER HAS INPUT PATIENT INFO IN A 
C  .CSV BLOCK FILE OR IN WORKING COPY FORMAT HAVE BEEN REMOVED. I.E.,
C  NOW ALL COVARIATES (INCLUDING WT AND CCR) WILL BE PART OF THE NADD
C  ADDITIONAL COVARIATES (SEE CODE IN EACH ROUTINE THAT SETS VALUES FOR
C  CV(I), I=1,NADD).

C-----------------------------------------------------------------------

C  TSTMULTH.FOR                                               DEC, 2010     
                          
C  TSTMULTH HAS THE FOLLOWING CHANGES TO TSTMULTG.FOR:                      
                          
C  1. THE USER IS TOLD IN SUBROUTINE SYMBOL THAT TO USE THIS MODEL FILE     
C  WITH A MODEL WHICH ASSUMES THE STANDARD 3-COMPARTMENT LINEAR MODEL       
C  AS USED BY THE LITTLE POPULATION PROGRAMS, HE MUST CODE N=-1. THIS       
C  WILL MEAN THAT THE id___ MODULES LINKED WITH bigmlt11.f (THE FIRST       
C  MAIN MODULE TO ALLOW THIS OPTION) WILL CALL ANAL3 RATHER THAN            
C  USERANAL IF N = -1. IN THIS CASE, HE WILL ALSO HAVE TO CODE IN THE       
C  PARAMETER EQUATIONS FOR KE,KA,KCP,KPC, AND V IN SUBROUTINE ANAL3. AND    
C  THEN THESE VALUES WILL BE AVAILABLE IN SUBROUTINE OUTPUT TO ESTABLISH    
C  Y(.).                  
                          
C  2. SUBROUTINES ANAL3, CASE1,... AS USED IN NPAG91.FOR/IDPC9A.FOR         
C  ARE NOW INCLUDED IN THIS MODULE. THEY ARE CHANGED A LITTLE TO BE         
C  COMPATIBLE WITH THE BIG POPULATION PROGRAMS. 

C  3. ALSO NOTE THAT IN SUBROUTINE GETIX, ADDITIONAL CODE IS REQUIRED
C  TO RESET ALL THE X(I) = 0 IN THE DEFAULT CASE WHEN N = -1.   
                          
C-----------------------------------------------------------------------    
                          
C  TSTMULTG.FOR                                               SEPT, 2009    
                          
C  TSTMULTG HAS THE FOLLOWING CHANGES TO TSTMULTF: 
                          
C  1. 3 NEW SUBROUTINES, GETFA, GETIX, AND GETTLAG, ARE ADDED TO THIS       
C  MODULE. EACH ROUTINE HAS A SECTION WHICH THE USER CAN EDIT TO SPECIFY    
C  EXPLICITLY THE VALUES FOR THE FA VECTOR, THE INITIAL CONDITION           
C  VECTOR, AND THEN TIMELAG VECTOR, RESPECTIVELY. THIS USER-SUPPLIED        
C  CODE MUST BE CONSISTENT, OF COURSE, WITH THE USER-DEFINED PARAMETERS     
C  (JUST AS THE CODE IN SUBROUTINES DIFFEQ AND OUTPUT MUST BE CONSISTENT    
C  WITH THE USER-DEFINED PARAMETERS).              
                          
C  2. BECAUSE OF 1., ALL REFERENCE TO IFA, IC, IVOL, NTLAG, AND IWRITE      
C  IN THIS MODULE HAVE BEEN REMOVED. THIS INCLUDES REMOVING NTLAG FROM      
C  COMMON/CNST.           
                          
C  3. NOTE THAT THIS TEMPLATE MODEL FILE IS FIRST USED WITH THE             
C  NPBIG15C.FOR PROGRAM, AND THE FIRST ID MODULES WHICH WILL CALL THE       
C  ROUTINES FROM THIS FILE ARE idm1x3.f, idm2x3.f, AND idm3x3.f, WHICH      
C  ARE PART OF THE bigmlt4.f "engine".             
                          
C-----------------------------------------------------------------------    
                          
C  TSTMULTF.FOR							8/6/09                       
                          
C  TSTMULTF HAS THE FOLLOWING CHANGES TO TSTMULTE: 
                          
C  1. COMMON/CNST2 IS INCLUDED IN SUBROUTINES DIFFEQ AND OUTPUT. THE        
C  REASON IS THAT THIS COMMON CONTAINS NDRUG, THE NO. OF DRUGS IN THE       
C  PATIENT DATA FILES, AND NADD, THE NO. OF ADDITIONAL COVARIATES. WITH     
C  THIS INFORMATION, DIFFEQ AND OUTPUT CAN IDENTIFY WHICH OF THE R(.)       
C  ARRAY ARE THE COVARIATES, AND LABEL THEM AS CV(1), ... CV(2 + NADD).     
                          
C  SIMILARY, THE NDRUG IV RATES WILL BE LABELED    
C  RATEIV(1), ..., RATEIV(NDRUG) IN DIFFEQ.        
                          
C  NOTE THAT THE REASON FOR IDENTIFYING CV(.) AND RATEIV(.) IS FOR          
C  CLARITY PURPOSES ONLY. 
                          
C  NOTE THAT RATEIV(7) AND CV(26) ARE ADDED TO THE DIMENSION STATEMENT.     
                          
C  NOTE THAT IN SUBROUTINE SYMBOL, IF THE USER NOW SELECTS IC(2) = -4       
C  (SEE EXAMPLE BELOW), IT WILL MEAN THAT THE INITIAL COMPARTMENT           
C  CONCENTRATION IN COMPARTMENT 2 IS NO LONGER RS(1,-IC(2)) = RS(1,4),      
C  BUT INSTEAD WILL BE THE 1ST VALUE OF COVARIATE NO. -IC(2) = 4.           
                          
                          
C  2. IN SUBROUTINE SYMBOL, NEW CODE ALLOWS THE USER TO SELECT A            
C  PARAMETER TO BE THE LOG OF A TIME LAG (THEN THE ID MODULES WILL          
C  CALL SUBROUTINE SHIFT WHICH WILL SET THIS TIMELAG EQUAL TO               
C  EXP(THIS PARAMETER VALUE).                      
                          
C  SIMILAR NEW CODE ALLOWS THE SAME OPTION FOR A PARAMETER TO BE THE        
C  LOG OF AN FA PARAMETER. AND THE ID MODULES WILL SET THE                  
C  CORRESPONDING FA PARAMETER = EXP(THIS PARAMETER VALUE).                  
                          
C-----------------------------------------------------------------------    
                          
C  TSTMULTE.FOR							5/28/09                      
                          
C  TSTMULTE HAS THE FOLLOWING CHANGES TO TSTMULTD THAT INITCN23 MADE TO     
C  INITCN22 (FOR THE SINGLE DRUG PROGRAM:          
                          
C  IT ALLOWS THE USER TO SELECT INITIAL COMPARTMENT AMOUNTS TO BE SET       
C  AS FUNCTIONS OF THE INITIAL COMPARTMENT CONCENTRATIONS (IN TSTMULTE,     
C  THE USER WAS LIMITED TO SETTING INITIAL COMPARTMENT AMOUNTS TO BE        
C  PARAMETERS ONLY).      
                          
C  THIS IS DONE AS FOLLOWS:                        
                          
C  IF THE USER SELECTS, E.G., IC(2) = -4, IT MEANS THAT THE INITIAL         
C  COMPARTMENT AMOUNT WILL NOT BE PARAMETER 4, BUT WILL BE CALCULATED       
C  BY THE ID MODULES WHICH WILL UNDERSTAND THAT RS(1,-IC(2)) IS TO BE       
C  THE INITIAL CONCENTRATION IN COMPARTMENT 2 (I.E., IF ANY IC(.) IS        
C  NEGATIVE, THE ID MODULES KNOW TO GET THE INITIAL CONCENTRATION FOR       
C  THE COMPARTMENT FROM THE 1ST COVARIATE VALUE OF THE COV. WHOSE INDEX     
C  IS -IC(2)). OF COURSE, THE VOLUMES OF ANY COMPARTMENTS WHOSE INITIAL     
C  AMOUNTS ARE TO BE SET THIS WAY WILL HAVE TO BE SUPPLIED TO THE ID        
C  MODULES AS WELL. THESE VALUES MUST BE DECLARED TO BE PARAMETERS, AND     
C  THE INDEX OF WHICH PARAMETERS THEY ARE WILL BE SUPPLIED VIA VECTOR       
C  IVOL(.), WHICH WILL BE PASSED ALONG WITH IC IN THE EXPANDED              
C  COMMON/INITCOND/IC,IVOL.                        
                          
C  SO IN THIS CASE, THE EQUATION IN THE ID MODULE FOR THE INITIAL           
C  COMPARTMENT AMOUNT FOR COMPARTMENT 2 WOULD BE CALCULATED AS              
C   X(2) = P(IVOL(2)) * RS(1, -IC(2))              
                          
C  MORE GENERALLY, THE CODE IN THE ID MODULES WOULD HAVE THE NEW            
C  EXPANDED COMMON/INITCOND/IC,IVOL, AND THEN WOULD HAVE THE LOGIC:         
                          
C	DO I=1,N                
C	 X(I) = 0.D0            
C	 IF(IC(I) .GT. 0) X(I) = P(IC(I))                
C	 IF(IC(I) .LT. 0) X(I) = P(IVOL(I)) * RS(1,-IC(I))                        
C	END DO                  
                          
C  NOTE THAT THIS TEMPLATE MODEL FILE IS COMPATIBLE WITH THE bigmlt2.f      
C  PROGRAM, INCLUDING bigmlt2.f, idm1x1.f, idm2x1.f, and idm3x1.f.          
                          
C-----------------------------------------------------------------------    
                          
C  TSTMULTD.FOR							6-13-02                      
                          
C  TSTMULTD IS A VARIATION OF TSTMULTC. TSTMULTC WAS USED WITH              
C  PATIENT FILES TESTM001.C,..., WHICH HAD THREE DRUGS, BUT NO TIMELAGS.    
C  TSTMULTD WILL BE USED WITH PATIENT FILES TESTM001.D,... WHICH ALSO       
C  HAVE 3 DRUGS, BUT THE TWO WITH BOLUS VALUES WILL HAVE TIMELAGS.          
C  NOTE THAT TSTMULTD/TESTM001.D,... WILL BE EQUIVALENT TO                  
C  TSTMULTC/TESTM001.C,... SINCE THE DOSAGE REGIMENT IN THE "D"             
C  SCENARIO WILL BE ADJUSTED SO THAT THE NET EFFECT AFTER THE TIMELAGS      
C  WILL BE EXACTLY THE SAME AS THE "C" SCENARIO.   
                          
C-----------------------------------------------------------------------    
                          
C  TSTMULTC IS A VARIATION OF TSTMULTB. TSTMULTB WAS USED WITH              
C  PATIENT FILES TESTM001.B,..., WHICH HAD TWO DRUGS. TSTMULTC              
C  WILL BE USED WITH PATIENT FILES TESTM001.C,... WHICH HAVE 3 DRUGS.       
C  NOTE THAT TSTMULTC/TESTM001.C,... WILL BE EQUIVALENT TO                  
C  TSTMULTB/TESTM001.B,... SINCE THE TOTAL IV'S AND THE TOTAL BOLUS         
C  AMOUNTS OF THE 3 DRUGS IN THE "C" SCENARIO  WILL BE THE SAME AS THE      
C  TOTALS IN THE "B" SCENARIO (AND ALL THE IV'S WILL GO INTO                
C  COMPARTMENT 2 AND ALL THE BOLI WILL GO INTO COMPARTMENT 1).              
                          
C-----------------------------------------------------------------------    
                          
C  TSTMULTB IS A VARIATION OF TSTMULTA. TSTMULTA WAS USED WITH              
C  PATIENT FILES TESTM001.A,..., WHICH ONLY HAD ONE DRUG. TSTMULTB          
C  WILL BE USED WITH PATIENT FILES TESTM001.B,... WHICH HAVE 2 DRUGS.       
C  NOTE THAT TSTMULTB/TESTM001.B,... WILL BE EQUIVALENT TO                  
C  TSTMULTA/TESTM001.A,... SINCE THE IV OF THE DRUG IN THE "A"              
C  SCENARIO WILL = DRUG 1 IN THE "B" SCENARIO, AND THE BOLUS OF THE         
C  DRUG IN THE "A" SCENARIO = DRUG 2 IN THE "B" SCENARIO.                   
                          
C-----------------------------------------------------------------------    
                          
C  TSTMULTA.FOR							6-12-02                      
                          
C  TSTMULTA IS A VERSION OF MULTDRUG.FOR. IT IS USED TO TEST THE            
C  NPBIG11.FOR/npbig8adapt.f PROGRAM, WHICH ALLOWS MULTIPLE DRUGS.          
C  IT HAS THE SAME MODEL AS TESTSING.FOR WHICH IS USED BY THE               
C  NPBIG10.FOR/npbig7adapt.f PROGRAM (WHICH DOESN'T ALLOW MULTIPLE          
C  DRUGS).                
                          
C-----------------------------------------------------------------------    
                          
C  MULTDRUG.FOR = INITCND3.FOR					5-14-02         
                          
C  MULTDRUG = INITCND3 HAS THE FOLLOWING CHANGES FROM INITCND2:             
                          
C  1. IT ALLOWS SETTING THE BOLUS COMPARTMENT NUMBERS,                      
C  NBCOMP(I),I=1,NDRUG IN SUBROUTINE SYMBOL. NBCOMP WILL NOW BE PASSED      
C  FROM SUBROUTINE SYMBOL VIA NEW COMMON/BOLUSCOMP TO SUBROUTINES FUNC,     
C  FUNC2, FUNC3 IN THE id____ MODULES (CURRENTLY idfix5g.f, idcy_53g.f,     
C  and idcy_63g.f).       
                          
C  2. NTLAG IN SUBROUTINE SYMBOL WILL NOW BE A VECTOR INSTEAD OF A          
C  SCALAR.                
C  NTLAG(I) = 0 IF DRUG I'S BOLUS COL. HAS NO TIMELAG PARAMETER;            
C             K IF DRUG I'S BOLUS COL. HAS TIMELAG AND IT'S PARAMETER       
C               NO K.     
                          
C  3. IFA IN SUBROUTINE SYMBOL WILL NOW BE A VECTOR INSTEAD OF A            
C  SCALAR.                
C  IFA(I) = 0 IF DRUG I WILL HAVE FA = 1.0.        
C           K IF DRUG I WILL HAVE AN FA WHOSE VALUE IS TO BE GIVEN          
C             BY PARAMETER K.                      
                          
                          
C-----------------------------------------------------------------------    
                          
C  INITCND2.FOR							4-23-02                      
                          
C  INITCND2 HAS THE FOLLOWING CHANGES FROM INITCOND.FOR:                    
                          
C  IT ALLOWS THE PARAMETER FA (FRACTION ABSORBED) TO BE SELECTED AS         
C  A RANDOM OR FIXED PARAMETER. THE VALUE OF FA WILL NOW BE PASSED FROM     
C  SUBROUTINE SYMBOL VIA NEW COMMON/FRABS TO SUBROUTINES FUNC, FUNC2,       
C  AND FUNC3 IN THE id___ MODULES (CURRENTLY idfix5f.f, idcy_53f.f,         
C  AND idcy_63f.f).       
                          
C-----------------------------------------------------------------------    
                          
C  INITCOND.FOR							1-21-00                      
                          
C  INITCOND HAS A STANDARD 3-COMPARTMENT MODEL, WITH compatibility          
c  for timelags (i.e., specifying a paramater as a timelag paramater),      
c  analytic solution(s) in SUBROUTINE OUTPUT, and the specification of      
c  one or more paramters in SUBROUTINE SYMBOL as initial conditions for     
c  the compartment amounts. It is compatible with the Big NPEM program,     
c  (npbig4.f, idfix5e.f, idcy_53e.f, idcy_63e.f, ...).                      
                          
c  Notes:                 
                          
c  0. This model file allows a timelag parameter, analytic solution(s)      
c     coded explicitly into SUBROUTINE OUTPUT, and initial compartment      
c     amounts being selected as parameters.        
c  1. P is dimensioned 32 instead of 25 in DIFFEQ and OUTPUT.               
c  2. Y is dimensioned 6 in OUTPUT.                
c  3. N is set = 0 in SYMBOL if analytic solutions are coded into           
c     SUBROUTINE OUTPUT.  
c  4. SUBROUTINE OUTPUT has an additional argument, T, the time at          
c     which analytic solutions (if they are desired) are to be found.       
c  5. SUBROUTINE SYMBOL has a new COMMON/INITCOND/IC, with IC               
c     dimensioned 20. The IC's are all be set = 0.0, except for those       
c     the user selects to be parameters.           
c  6. See SYMBOL notes to see how to set a Timelag paramater.               
                          
C-----------------------------------------------------------------------    
                          
      SUBROUTINE DIFFEQ(NDIM,T,X,XP,RPAR,IPAR)     
      IMPLICIT REAL*8(A-H,O-Z) 
      COMMON /PARAMD/ P   
      COMMON /INPUT/ R,B  
      COMMON /DESCR/ AGE,HEIGHT,ISEX,IETHFLG       
      COMMON /CNST2/ NPL,NUMEQT,NDRUG,NADD
      DIMENSION X(NDIM),XP(NDIM),P(32),R(37),B(20),CV(26),RATEIV(7)         
                          
                          
C*********** ESTABLISH IV AND COVARIATE VALUES BELOW: ******************    
                          
C  THE NDRUG IV VALUES ARE IN R(1), R(3), ..., R(2*NDRUG - 1). THE          
C  NDRUG BOLUS VALUES ARE IN R(2), R(4), ..., R(2*NDRUG) AND ARE            
C  UNNEEDED IN THIS ROUTINE.                       
                          
	DO I = 1,NDRUG           
	 RATEIV(I) = R(2*I - 1)  
	END DO                   
                          
C  NOW, RATEIV(I) IS THE IV RATE FOR DRUG I, I=1,7.
                          
                          
C  AS INDICATED ABOVE, THE FIRST 2*NDRUG VALUES IN R ARE THE (IV,BOLUS)     
C  VALUES FOR EACH OF THE NDRUG DRUGS, IN ORDER. THE FOLLOWING NADD
C  VALUES ARE THE ADDITIONAL COVARIATES IN EACH PATIENT'S WORKING COPY
C  PATIENT DATA FILE (BEYOND THE PERMANENT 4 AT THE TOP OF EACH FILE,
C  I.E., AGE, HEIGHT, ISEX, IETHFLG AS PASSED IN COMMON /DESCR). 

C  I.E., THE FIRST COVARIATE VALUE WILL BE IN ENTRY 2*NDRUG + 1, AND
C  THE LAST COVARIATE VALUE WILL BE IN ENTRY 2*NDRUG + NADD.
                          
	DO I = 1, NADD       
	 CV(I) = R(2*NDRUG + I)  
	END DO                   
                          
C  NOW CV(1), ..., CV(NADD) ARE THE NADD COVARIATE VALUES.          
                          
                          
C*********** ESTABLISH IV AND COVARIATE VALUES ABOVE: ******************    

                          
      RETURN              
      END                 
                          
C-----------------------------------------------------------------------    
                          
      SUBROUTINE OUTPUT(T,Y)                       
      IMPLICIT REAL*8(A-H,O-Z)                     
      COMMON /PARAMD/ P   
      COMMON /STATE/ X    
      COMMON /INPUT/ R,B  
      COMMON /DESCR/ AGE,HEIGHT,ISEX,IETHFLG       
      COMMON /CNST2/ NPL,NUMEQT,NDRUG,NADD         
      DIMENSION X(20),P(32),Y(6),R(37),B(20),CV(26)
                                                                             
C*********** ESTABLISH COVARIATE VALUES BELOW: ******************           
                          
	DO I = 1, NADD       
	 CV(I) = R(2*NDRUG + I)  
	END DO                   
                          
C  NOW CV(1), ..., CV(NADD) ARE THE NADD COVARIATE VALUES.          
C  SEE COMMENTS IN SUBROUTINE DIFFEQ FOR DETAILS.  
                          
C*********** ESTABLISH COVARIATE VALUES ABOVE: ******************           

 
                          
C  EX: Y(1) = X(2)/P(2)  
     
       V = P(2)    
       Y(1) = X(2)/V
       Y(2) = X(3)
                          
       RETURN             
       END                
                          
C-----------------------------------------------------------------------    
                          
      SUBROUTINE SYMBOL   
      IMPLICIT REAL*8(A-H,O-Z)                     
      CHARACTER PSYM(32)*11                        
      COMMON /CNST/ N,ND,NI,NUP,NUIC,NP            
      COMMON/BOLUSCOMP/NBCOMP                      
      DIMENSION NBCOMP(7) 
                          
C***************** SET BOLUS COMPARTMENT NUMBERS *****************          
                          
C  NBCOMP(I) IS THE COMPARTMENT NO. FOR DRUG I'S BOLUS INPUT.               
C  THE DEFAULT IS THAT NBCOMP(I) = I, I=1,7 (NOTE THAT 7 IS THE MAXIMUM     
C  NO. OF DRUGS). SO, FOR EXAMPLE, IF THERE IS 1 DRUG, ITS BOLUS WILL       
C  BY DEFAULT GO INTO COMPARTMENT NO. 1.           
                          
	DO I = 1,7               
	 NBCOMP(I) = I           
	END DO                   
                          
C  REASSIGN THE BOLUS COMPARTMENT NOS. AS DESIRED HERE ...                  
                          
C	NBCOMP(2) = 1            
C	NBCOMP(3) = 1            
                          
C  NOTE THAT THE BOLI FOR DRUGS 2 AND 3 ARE REASSIGNED TO GO INTO           
C  COMPARTMENT 1.         
                          
                          
                          
C******* SET FA (FRACTION ABSORBED) BY COMPLETING SUBROUTINE GETFA *****    
                          
C  COMPLETE SUBROUTINE GETFA IN THIS MODULE TO ESTABLISH THE FORMULAS       
C  FOR EACH DRUG'S FA.    
                          
C******* SET FA (FRACTION ABSORBED) BY COMPLETING SUBROUTINE GETFA *****    
                          
                          
C********** SET THE TIMELAGS BY COMPLETING SUBROUTINE GETTLAG **********    
                          
C  COMPLETE SUBROUTINE GETTLAG IN THIS MODULE TO ESTABLISH THE FORMULAS     
C  FOR EACH DRUG'S TIMELAG.                        
                          
C********** SET THE TIMELAGS BY COMPLETING SUBROUTINE GETTLAG **********    
                          
                          
C**** SET INITIAL COMPARTMENT VALUES BY COMPLETING SUBROUTINE GETIX ****    
                          
C  COMPLETE SUBROUTINE GETIX IN THIS MODULE TO ESTABLISH THE FORMULAS       
C  FOR EACH COMPARTMENT'S INITIAL AMOUNT.          
                          
C**** SET INITIAL COMPARTMENT VALUES BY COMPLETING SUBROUTINE GETIX ****    
                          
                          
                          
                          
C************************* SET N BELOW: ********************************    
                          
C  N IS THE NO. OF COMPARTMENTS IN THE MODEL (MAX. NO. = 20).               
C  THIS SHOULD BE THE NUMBER OF DIFFERENTIAL EQUATIONS YOU HAVE CODED       
C  INTO SUBROUTINE DIFFEQ, EXCEPT ...              
                          
C  1. SET    N=0   IF THE OUTPUT EQUATION(S) FOR Y( ) IS(ARE) CODED         
C  EXPLICITLY INTO SUBROUTINE OUTPUT (I.E., SUBROUTINE DIFFEQ WILL NOT      
C  BE USED), OR           
                          
C  2. SET    N=-1 IF SUBROUTINE DIFFEQ WILL NOT BE USED BECAUSE THE         
C  MODEL IS ASSUMED TO BE THE STANDARD 3-COMPARTMENT LINEAR MODEL. IN       
C  THIS CASE, THE id___ MODULES LINKED WITH THIS MODEL FILE WILL CALL       
C  SUBROUTINE ANAL3 (SAME AS IN THE "LITTLE" NPAG PROGRAM) RATHER THAN      
C  USERANAL (WHICH IS WHY DIFFEQ WILL NOT BE CALLED). AND IN THIS CASE,     
C  YOU MUST CODE IN THE EQUATIONS FOR KE, KA, KCP, KPC, AND V IN            
C  SUBROUTINE ANAL3 IN THIS MODULE (AS FUNCTIONS OF THE PARAMETERS OF       
C  YOUR MODEL). THEN YOU WILL BE ABLE TO USE THESE VALUES IN YOUR CODE      
C  IN SUBROUTINE OUTPUT ABOVE.                     
                          
C  NOTE THAT N=x MUST BE IN COLUMNS 8 - 10 IF x IS 0 TO 9, AND IN           
C  COLUMNS 8 - 11 IF x IS 10 - 20 OR -1.           
                          
       N=-1                
                          
C************************* SET N ABOVE: ********************************    
                          
                          
                          
C**** SET THE NUMBER OF AND THE NAMES OF THE PARAMATERS *****               
                          
C  SET    NP    EQUAL TO THE NUMBER OF PARAMATERS. THE MAX. NO. OF          
C  PARAMETERS IS 32. SET PSYM(I) EQUAL TO THE NAME OF PARAMETER I.          
                          
C  NOTE THAT   NP=J   MUST BE IN COLUMNS 8 THROUGH 11 (8 THROUGH 12 IF      
C  J IS A 2 DIGIT NUMBER).
                          
C  THE NAMES OF THE PARAMATERS MUST BE SET AS SHOWN BELOW ...               
C  PSYM(I)=' MUST START IN COLUMN 8, THE NAME MUST BE NO MORE THAN          
C  11 CHARACTERS, AND THERE MUST BE NO SPACES ANYWHERE IN THE STATEMENT.    
                          
       NP=5               
       PSYM(1)='KE'      
       PSYM(2)='V'      
       PSYM(3)='KA'
       PSYM(4)='KCP'
       PSYM(5)='KPC'
                     
                          
                          
       RETURN             
       END                
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC              
C                         
        SUBROUTINE GETFA(FA)                       
        IMPLICIT REAL*8(A-H,O-Z) 
        COMMON /PARAMD/ P 
        COMMON /INPUT/ R,B
        COMMON /DESCR/ AGE,HEIGHT,ISEX,IETHFLG     
        COMMON /CNST2/ NPL,NUMEQT,NDRUG,NADD
        DIMENSION P(32),R(37),B(20),CV(26),FA(7)   
                         
C*********** ESTABLISH COVARIATE VALUES BELOW: ******************           
                          
	DO I = 1, NADD       
	 CV(I) = R(2*NDRUG + I)  
	END DO                   
                          
C  NOW CV(1), ..., CV(NADD) ARE THE NADD COVARIATE VALUES.          
C  SEE COMMENTS IN SUBROUTINE DIFFEQ FOR DETAILS.  
                          
C*********** ESTABLISH COVARIATE VALUES ABOVE: ******************           
 
                         
                          
C  THIS ROUTINE IS CALLED BY THE ID ROUTINES TO ESTABLISH THE VALUES        
C  FOR FA(ID), ID=1,NDRUG, FOR THIS MODEL.         
                          
                          
C  THE DEFAULT IS THAT NONE OF THE PARAMETERS DEFINED IN SUBROUTINE         
C  SYMBOL IS AN FA PARAMETER, WHICH MEANS THAT ALL FA VALUES ARE            
C  1.0.                   
                          
	DO I = 1,NDRUG           
	 FA(I) = 1.D0            
	END DO                   
                          
                          
C********************** USER DEFINED CODE BELOW ************************    
                          
C  FOR EACH DRUG WHOSE FA VALUE IS NOT 1.0, WRITE IN THE CODE BELOW FOR     
C  THAT FA. EACH FA VALUE CAN BE A CONSTANT, OR A FUNCTION OF A             
C  PARAMETER AND/OR A COVARIATE.                   
                          
C  FOR EXAMPLE, IF THE FA FOR DRUG 3 IS TO BE      
C  EXP(PARAMETER 7) x THE 5TH COVARIATE VALUE, THEN THE EQUATION IS:        
                          
C     FA(3) = DEXP(P(7)) * CV(5)                   
                                                    
                                                    
C********************** USER DEFINED CODE ABOVE ************************    
                          
                          
                          
	RETURN                   
	END                      
	                         
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC              
C                         
        SUBROUTINE GETIX(N,X)                      
        IMPLICIT REAL*8(A-H,O-Z)
        COMMON /PARAMD/ P 
        COMMON /INPUT/ R,B
        COMMON /DESCR/ AGE,HEIGHT,ISEX,IETHFLG     
        COMMON /CNST2/ NPL,NUMEQT,NDRUG,NADD 
        DIMENSION P(32),R(37),B(20),CV(26),X(20)   
                         
C*********** ESTABLISH COVARIATE VALUES BELOW: ******************           
                          
	DO I = 1, NADD       
	 CV(I) = R(2*NDRUG + I)  
	END DO                   
                          
C  NOW CV(1), ..., CV(NADD) ARE THE NADD COVARIATE VALUES.          
C  SEE COMMENTS IN SUBROUTINE DIFFEQ FOR DETAILS.  
                          
C*********** ESTABLISH COVARIATE VALUES ABOVE: ******************           
                          
                          
C  THIS ROUTINE IS CALLED BY THE ID ROUTINES TO ESTABLISH THE VALUES        
C  FOR THE INITIAL COMPARTMENT AMOUNTS, X(I), I=1,N, FOR THIS MODEL.        
                          
C  THE DEFAULT IS THAT NONE OF THE PARAMETERS DEFINED IN SUBROUTINE         
C  SYMBOL IS AN INITIAL COMPARTMENT PARAMETER, WHICH MEANS THAT ALL         
C  X VALUES ARE SET = 0.  
              
        IF(N .GT. 0) THEN            
         DO I = 1,N               
          X(I) = 0.D0             
         END DO  
        ENDIF

C  BUT NOTE THAT IF THE USER HAS SET N = -1 IN SYMBOL (WHICH MEANS HIS
C  MODEL IS THE STANDARD 3-COMPARTMENT LINEAR MODEL), THE DO LOOP ABOVE
C  WILL NOT BE USED. THE FOLLOWING CODE TAKES CARE OF THIS SITUATION:

        IF(N .EQ. -1) THEN
	   DO I = 1,3               
	    X(I) = 0.D0             
	   END DO 
        ENDIF 

                                                  
C********************** USER DEFINED CODE BELOW ************************    
                          
                          
C  EX: TO SET THE INITIAL AMOUNT OF COMPARTMENT 2 = PARAMETER 3, USE:       
                          
C       X(2) = P(3)       
                          
C  EX: TO SET THE INITIAL AMOUNT OF COMPARTMENT 2 ASSUMING:                 
C      i. THE INITIAL CONCENTRATION IN COMPARTMENT 2 IS COVARIATE NO. 3;    
C      ii. VS FOR COMPARTMENT 2 IS PARAMETER NO. 4; AND                     
C      iii. THE WEIGHT OF THE SUBJECT IS COVARIATE 5, USE THE CODE:         
                          
C      X(2) = CV(3) * P(4) * CV(5)                 
                                                                                                     
                          
C****************** SET INITIAL CONDITIONS ABOVE ********************       
                          
                          
                          
	RETURN                   
	END                      
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC              
C                         
        SUBROUTINE GETTLAG(TLAG)                   
        IMPLICIT REAL*8(A-H,O-Z)
        COMMON /PARAMD/ P 
        COMMON /INPUT/ R,B
        COMMON /DESCR/ AGE,HEIGHT,ISEX,IETHFLG     
        COMMON /CNST2/ NPL,NUMEQT,NDRUG,NADD 
        DIMENSION P(32),R(37),B(20),CV(26),TLAG(7) 
                          
                         
C*********** ESTABLISH COVARIATE VALUES BELOW: ******************           
                          
	DO I = 1, NADD       
	 CV(I) = R(2*NDRUG + I)  
	END DO                   
                          
C  NOW CV(1), ..., CV(NADD) ARE THE NADD COVARIATE VALUES.          
C  SEE COMMENTS IN SUBROUTINE DIFFEQ FOR DETAILS.  
                          
C*********** ESTABLISH COVARIATE VALUES ABOVE: ******************           

                          
C  THIS ROUTINE IS CALLED BY THE ID ROUTINES TO ESTABLISH THE VALUES        
C  FOR TLAG(ID), ID=1,NDRUG, FOR THIS MODEL.       
                          
                          
C  THE DEFAULT IS THAT NONE OF THE PARAMETERS DEFINED IN SUBROUTINE         
C  SYMBOL IS A TLAG PARAMETER, WHICH MEANS THAT ALL TLAG VALUES ARE 0.      
                          
	DO I = 1,NDRUG           
	 TLAG(I) = 0.D0          
	END DO                   
                          
                          
C********************** USER DEFINED CODE BELOW ************************    
                          
C  FOR EACH DRUG WHICH HAS A TIMELAG INTO ITS BOLUS COMPARTMENT, WRITE      
C  IN THE CODE BELOW FOR THAT TIMELAG VALUE, TLAG(.). EACH TLAG(I) VALUE    
C  CAN BE A CONSTANT, OR A FUNCTION OF A PARAMETER AND/OR A COVARIATE.      
                          
C  FOR EXAMPLE, IF PARAMETER 8 IS THE LOG OF THE TIMELAG PARAMETER FOR      
C  DRUG 4, SET:           
                          
C  TLAG(4) = DEXP(P(8))   
                                                                                                        
                          
C********************** USER DEFINED CODE ABOVE ************************    
                          
                          
                          
	RETURN                   
	END                      
                          
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC              
C                         
        SUBROUTINE ANAL3(X,TIN,TOUT)               
C                         
C  PURPOSE: GIVEN X(TIN) THE PROGRAM CALCULATES X(TOUT), WHERE X            
C           IS THE STATE VECTOR FOR A TWO COMPARTMENT LINEAR MODEL          
C           WITH FIRTST ORDER ABSORBTION AND INTRAVENOUS INPUTS.            
C                         
C  ARGUMENTS ON INPUT:    
C           X - AN ARRAY. X(I), I=1,3, SHOULD      
C               BE SET TO THE AMOUNT OF DRUG IN THE ABSORBTION,             
C               CENTRAL, AND PERIPHERAL COMPARTMENTS, RESPECTIVELY,         
C               AT TIME T=TIN. NOTE, HOWEVER, TO BE COMPATIBLE WITH         
C               THIS PROGRAM, X IS DIMENSIONED 20 IN THIS ROUTINE.          
C         TIN - CURRENT VALUE OF TIME.             
C        TOUT - TIME AT WHICH SOLUTION IS DESIRED. 
C                         
C  	KE,KA,KCP,KPC FROM COMMON/RATESV FROM SUBROUTINE PARDEF                 
	                         
C  ARGUMENTS ON OUTPUT:   
C           X - THE COMPARTMENT AMOUNTS AT T=TOUT. 
C         TIN - SET AT TOUT                        
C                         
        IMPLICIT REAL*8(A-H,O-Z)                   
        DOUBLE PRECISION KE,KA,KCP,KPC             
        DIMENSION X(20),P(32),R(37),B(20),CV(26) 
        COMMON /CNST2/ NPL,NUMEQT,NDRUG,NADD                  
        COMMON /RATESV/ KE,KA,KCP,KPC,V            
        COMMON /INPUT/ R,B
        COMMON /PARAMD/ P   

C*********** ESTABLISH COVARIATE VALUES BELOW: ******************           
                          
	DO I = 1, NADD       
	 CV(I) = R(2*NDRUG + I)  
	END DO                   
                          
C  NOW CV(1), ..., CV(NADD) ARE THE NADD COVARIATE VALUES.          
C  SEE COMMENTS IN SUBROUTINE DIFFEQ FOR DETAILS.  
                          
C*********** ESTABLISH COVARIATE VALUES ABOVE: ******************           
     
                     
                          
C********************** USER DEFINED CODE BELOW ************************    
                          
C  IF THE USER CODED IN N=-1 IN SUBROUTINE SYMBOL ABOVE, HE MUST            
C  ESTABLISH THE 5 BASIC PARAMETERS {KE,KA,KCP,KPC,V} OF THE                
C  STANDARD 3-COMPARTMENT MODEL HERE.              
                          
C  THE FOLLOWING IS AN EXAMPLE:                    
                          
C  KE = DEXP(P(1))        
C  KA = DEXP(P(2))        
C  KCP = DEXP(P(3))       
C  KPC = DEXP(P(4))       
C  V = DEXP(P(5))         
                          
      KE = P(1)
      V = P(2)
      KA = P(3)
      KCP = P(4)
      KPC = P(5)
                          
                          
C********************** USER DEFINED CODE ABOVE ************************    
                          
                          
        T=TOUT-TIN        
C                         
C  SELECT WHICH CASE THE MODEL CONFORMS TO.        
C                         
        IF(KCP.EQ.0.0D0.AND.KPC.EQ.0.0D0) THEN     
          IF(KA.EQ.0.0D0) ICASE=1                  
          IF(KA.NE.0.0D0) ICASE=2	                 
                          
	ELSE                     
                          
	  IF(KA.EQ.0.0D0) ICASE=3
          IF(KA.NE.0.0D0) ICASE=4                  
                          
	ENDIF                    
                          
        GO TO (100,200,300,400), ICASE             
C                         
C  CASE 1 SOLUTION - 1 COMP. NO 1ST ORDER INPUT    
C                         
100     CALL CASE1(X(2),T)
	TIN=TOUT                 
        RETURN            
C                         
C  CASE 2 SOLUTION - 1 COMP. + 1ST ORDER INPUT.    
C                         
200     CALL CASE2(X(1),X(2),T)                    
        TIN=TOUT          
        RETURN            
C                         
C  CASE 3 SOLUTION - 2 COMPARTMENT NO 1ST ORDER INPUT.                      
C                         
300     CALL CASE3(X(2),X(3),T)                    
        TIN=TOUT          
        RETURN            
C                         
C CASE 4 SOLUTION - 2 COMP. + 1ST ORDER INPUT.     
C                         
C400    CONTINUE          
400     CALL CASE4(X,T)   
        TIN=TOUT          
        RETURN            
        END               
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC              
C                         
        SUBROUTINE CASE1(X2,T)                     
C                         
C  SOLUTION FOR 1 COMP. MODEL WITH NO 1ST ORDER INPUT.                      
C                         
        IMPLICIT REAL*8(A-H,O-Z)                   
        DOUBLE PRECISION KE,KA,KCP,KPC                                     
        DIMENSION R(37),B(20)                      
        COMMON /RATESV/ KE,KA,KCP,KPC,V            
        COMMON /INPUT/ R,B
                          
C                         
C*****SUBCASE 1A - KE=0.0*****                     
C                         
        IF(KE.NE.0.0D0) GO TO 10                   
        X2=T*R(1)+X2      
        RETURN            
C                         
C*****COMPLETE SOLUTION*****                       
C                         
10      EKET=DEXP(-KE*T)  
        X2=R(1)*(1.0D0-EKET)/KE+X2*EKET            
C                         
        RETURN            
        END               
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC              
C                         
        SUBROUTINE CASE2(X1,X2,T)                  
C                         
C  SOLUTION FOR 1 COMP. MODEL WITH 1ST ORDER INPUT.
                       
        IMPLICIT REAL*8(A-H,O-Z)                   
        DOUBLE PRECISION KE,KA,KCP,KPC             
        COMMON /RATESV/ KE,KA,KCP,KPC,V            
        COMMON /INPUT/ R,B
        DIMENSION R(37),B(20)                      
                          
C                         
C*****SUBCASE 2A - KA=KE*****                      
C                         
        IF(KA.NE.KE) GO TO 30                      
        EKT=DEXP(-KE*T)   
        X2=(X2-R(1)/KE)*EKT+R(1)/KE+KE*X1*T*EKT    
        X1=X1*DEXP(-KA*T) 
        RETURN            
C                         
C*****SUBCASE 2B - KE=0.0*****                     
C                         
30      IF(KE.NE.0.0D0) GO TO 50                   
        EKAT=DEXP(-KA*T)  
        X2=X2+T*R(1)+X1*(1.0D0-EKAT)               
        X1=X1*EKAT        
        RETURN            
C                         
C*****COMPLETE SOLUTION*****                       
C                         
50      EKET=DEXP(-KE*T)  
        EKAT=DEXP(-KA*T)  
        X2=X2*EKET+R(1)*(1.0D0-EKET)/KE+           
     X    KA*X1*(EKET-EKAT)/(KA-KE)                
        X1=X1*EKAT        
C                         
        RETURN            
        END               
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC              
C                         
        SUBROUTINE CASE3(X2,X3,T)                  
C                         
C  SOLUTION FOR 2 COMP. MODEL WITH NO 1ST ORDER INPUT.                      
C                         
        IMPLICIT REAL*8(A-H,O-Z)                   
        DOUBLE PRECISION KE,KA,KCP,KPC,L1,L2       
        COMMON /RATESV/ KE,KA,KCP,KPC,V            
        COMMON /INPUT/ R,B
        DIMENSION EA(2,2),R(37),B(20)              
                          
C                         
C  CALCULATE EIGEN VALUES L1 AND L2.               
C                         
        T1=KE+KCP+KPC     
        T2=DSQRT(T1*T1-4.0D0*KE*KPC)               
        L1=0.5D0*(T1+T2)  
        L2=0.5D0*(T1-T2)  
C                         
C*****SUBCASE 3A - L2=0.0*****                     
C                         
        IF(L2.NE.0.0D0) GO TO 200                  
        EL1T=DEXP(-L1*T)  
C                         
C  FORM EXP(AT)*L1        
        OEL1T=1.0D0-EL1T  
        EA(1,1)=(L1-KPC)*EL1T+KPC                  
        EA(1,2)=KPC*OEL1T 
        EA(2,1)=KCP*OEL1T 
        EA(2,2)=(L1-KE-KCP)*EL1T+KE+KCP            
C                         
C  FORM THE PARTICULAR SOLUTION*L1                 
        P1=R(1)*((1.0D0-KPC/L1)*OEL1T+KPC*T)       
        P2=R(1)*((-KCP/L1)*OEL1T+KCP*T)            
C                         
C  FORM COMPLETE SOLUTION 
        C1=EA(1,1)*X2+EA(1,2)*X3+P1                
        C2=EA(2,1)*X2+EA(2,2)*X3+P2                
        X2=C1/L1          
        X3=C2/L1          
        RETURN            
C                         
C*****COMPLETE SOLUTION*****                       
C                         
200     CONTINUE          
        EL1T=DEXP(-L1*T)  
        EL2T=DEXP(-L2*T)  
        OEL1T=1.0D0-EL1T  
        OEL2T=1.0D0-EL2T  
        DEL2L1=EL2T-EL1T  
C                         
C  FORM EXP(AT)*(L1-L2)   
        EA(1,1)=(L1-KPC)*EL1T+(KPC-L2)*EL2T        
        EA(1,2)=KPC*DEL2L1
        EA(2,1)=KCP*DEL2L1
        EA(2,2)=(L1-KE-KCP)*EL1T+(KE+KCP-L2)*EL2T  
C                         
C  FORM THE PARTICULAR SOLUTION*(L1-L2).           
        P1=R(1)*((1.0D0-KPC/L1)*OEL1T+(KPC/L2-1.0D0)*OEL2T)                 
        P2=R(1)*((-KCP/L1)*OEL1T+(KCP/L2)*OEL2T)   
C                         
C  FORM THE COMPLETE SOLUTION                      
        D=L1-L2           
        C1=EA(1,1)*X2+EA(1,2)*X3+P1                
        C2=EA(2,1)*X2+EA(2,2)*X3+P2                
        X2=C1/D           
        X3=C2/D           
        RETURN            
        END               
C                         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCŠC            
        SUBROUTINE CASE4(X,T)                      
C                         
C  SOLUTION FOR 2 COMP. MODEL WITH 1ST ORDER INPUT.
C                         
        IMPLICIT REAL*8(A-H,O-Z)                   
        DOUBLE PRECISION KE,KA,KCP,KPC,L1,L2       
        COMMON /RATESV/ KE,KA,KCP,KPC,V            
        COMMON /INPUT/ R,B
        DIMENSION EA(2,2),R(37),B(20),X(20)        
                          
                          
C  CALCULATE EIGEN VALUES L1 AND L2.               
                          
        T1=KE+KCP+KPC     
        T2=DSQRT(T1*T1-4.0D0*KE*KPC)               
        L1=0.5D0*(T1+T2)  
        L2=0.5D0*(T1-T2)  
C                         
C*****SUBCASE 4A - L2=0.0*****                     
C                         
        IF(L2.NE.0.0D0) GO TO 200                  
        EL1T=DEXP(-L1*T)  
C                         
C  FORM EXP(AT)*L1        
        OEL1T=1.0D0-EL1T  
        EA(1,1)=(L1-KPC)*EL1T+KPC                  
        EA(1,2)=KPC*OEL1T 
        EA(2,1)=KCP*OEL1T 
        EA(2,2)=(L1-KE-KCP)*EL1T+KE+KCP            
C                         
C  FORM THE PARTICULAR SOLUTION*L1 FROM IV         
        P1A=R(1)*((1.0D0-KPC/L1)*OEL1T+KPC*T)      
        P2A=R(1)*((-KCP/L1)*OEL1T+KCP*T)           
C                         
C  FORM THE PARTICULAR SOLUTION*L1 FROM 1ST ORDER ABS. INPUT.               
        EKAT=DEXP(-KA*T)  
        RL1=(EL1T-EKAT)/(KA-L1)                    
        RKA=(1.0D0-EKAT)/KA                        
        P1B=KA*X(1)*((L1-KPC)*RL1+KPC*RKA)         
        P2B=KA*X(1)*(-KCP*RL1+KCP*RKA)             
C                         
C  FORM COMPLETE SOLUTION 
        C1=EA(1,1)*X(2)+EA(1,2)*X(3)+P1A+P1B       
        C2=EA(2,1)*X(2)+EA(2,2)*X(3)+P2A+P2B       
        X(1)=X(1)*EKAT    
        X(2)=C1/L1        
        X(3)=C2/L1        
        RETURN            
C                         
C*****SOLUTIONS FOR THE FOLLOWING CASES:*****      
C           1. SUB CASE 4B - KA=L1                 
C           2. SUB CASE 4C - KA=L2
C           3. COMPLETE SOLUTION.        
C                         
200     CONTINUE          
C                         
C---FORM PART OF SOLUTION COMMON TO ALL 3 CASES.---
        EL1T=DEXP(-L1*T)  
        EL2T=DEXP(-L2*T)  
        EKAT=DEXP(-KA*T)  
        OEL1T=1.0D0-EL1T  
        OEL2T=1.0D0-EL2T  
        DEL2L1=EL2T-EL1T  
C                         
C  FORM EXP(AT)*(L1-L2)   
        EA(1,1)=(L1-KPC)*EL1T+(KPC-L2)*EL2T        
        EA(1,2)=KPC*DEL2L1
        EA(2,1)=KCP*DEL2L1
        EA(2,2)=(L1-KE-KCP)*EL1T+(KE+KCP-L2)*EL2T  
C                         
C  FORM THE PARTICULAR SOLUTION*(L1-L2) FROM THE INFUSION(A).               
        P1A=R(1)*((1.0D0-KPC/L1)*OEL1T+(KPC/L2-1.0D0)*OEL2T)                
        P2A=R(1)*((-KCP/L1)*OEL1T+(KCP/L2)*OEL2T)  
C                         
C---FORM THE PARTICULAR SOLUTION*(L1-L2) FROM 1ST ORDER INPUT,              
C                                   FOR THE APPROPRIATE  CASE(B).---        
C  SUBCASE 4B             
        IF(KA.NE.L1) GO TO 240                     
        RL=DEL2L1/(L1-L2) 
        P1B=L1*X(1)*(T*(L1-KPC)*EL1T+(KPC-L2)*RL)  
        P2B=L1*X(1)*(-T*KCP*EL1T+KCP*RL)           
        GO TO 300         
C                         
C  SUBCASE 4C             
240     IF(KA.NE.L2) GO TO 280                     
        RL=DEL2L1/(L1-L2) 
        P1B=L2*X(1)*((L1-KPC)*RL+T*(KPC-L2)*EL2T)  
        P2B=L2*X(1)*(-KCP*RL+T*KCP*EL2T)           
        GO TO 300         
C                         
C  COMPLETE SOL SUBCASE   
280     RL1KA=(EL1T-EKAT)/(KA-L1)                  
        RL2KA=(EL2T-EKAT)/(KA-L2)                  
        P1B=KA*X(1)*((L1-KPC)*RL1KA+(KPC-L2)*RL2KA)
        P2B=KA*X(1)*(-KCP*RL1KA+KCP*RL2KA)         
C                         
C---FORM THE COMPOSITE SOLUTION---                 
300     D=L1-L2           
        C1=EA(1,1)*X(2)+EA(1,2)*X(3)+P1A+P1B       
        C2=EA(2,1)*X(2)+EA(2,2)*X(3)+P2A+P2B       
        X(1)=X(1)*EKAT    
        X(2)=C1/D         
        X(3)=C2/D         
        RETURN            
        END               
                          
