# Automated Prediction of the Response to Neoadjuvant Chemoradiotherapy in Patients Affected by Rectal Cancer

2022 MDPI 논문



### Proposal

- Propose a novel automated pipeline for the segmentation of MRI scans of patients with LARC in order to predict the response to nCRT using radiomic features.

CRT : Chemoradiotherapy의 약어(화학방사선요법|암치료)

LARC : Locally Advanced Rectal Cancer의 약어(장의 지역 진행성 암)

직장(Rectum) 암이 주변 조직이나 림프절에 확산하거나 주변 조직에 침범한 상태를 나타냅니다. "Locally Advanced"는 종양이 원래 발생한 장 부위를 벗어나지 않았지만 주변 조직에 확산되거나 침범한 상태를 의미

- 후향적 분석이 포함됨(43명 환자의 T2-MRI )
- The analysis of radiomic features allowed us to predict the TRG score, which agreed with the state-of-the-art results.

### Analysis Method

1. 자동으로 식별된 병변 부위에 대해 radiomics feature 100개 추출.(각 환자당)

2. PCA(주성분분석)을 통해 co-linearities, non-informative components 제거

​	PCA를 적용해 전체 분산의 90%를 대표하는 6개의 구성 요소를 찾음

3. SVC 이용해 분석

4. 10 fold 교차 검증

   평가 지표 : MCC(Matthews Correlation Coeffiecient)



### Results

- Segmentation result

​	![image-20231207105609978](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20231207105609978.png)



- Final result

​	![image-20231207105711198](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20231207105711198.png)



----

Clinic data와 결합해서 분석하지는 않음

Clinic data에서 GT값인 TRG만 가져와서 분석에 사용함