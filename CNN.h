//*****************************************************************************************//
//**                                                                                     **//
//**                                    CNN                                              **//
//**                                                                                     **//
//*****************************************************************************************//
#ifndef Class_CNN_Header
#define Class_CNN_Header

#include "../Common/Direct3DWrapperNN/Dx_NN.h"
#include "../CreateGeometry/CreateGeometry.h"
#include "../Common/Direct3DWrapper/Dx12Process.h"

class CNN;
class Convolution;
class Pooling;

class GradCAM :public DxGradCAM {

protected:
	PolygonData2D dgc;

public:
	GradCAM(UINT srcWid, UINT srcHei, UINT SizeFeatureMapW, UINT SizeFeatureMapH, UINT NumGradientEl,
		UINT NumFil, UINT inputsetnum);
	void Draw(float x, float y);
};

class Affine :public DxNeuralNetwork {

protected:
	friend CNN;
	Convolution* errCn = nullptr;
	Pooling* errPo = nullptr;
	Convolution* inCn = nullptr;
	Pooling* inPo = nullptr;
	PolygonData2D dnn;
	UINT NumFilter;
	std::unique_ptr<PolygonData> pdArr = nullptr;
	std::unique_ptr<std::unique_ptr<CoordTf::VECTOR3[]>[]> nPos = nullptr;
	std::unique_ptr<float[]> nAngleY = nullptr;
	std::unique_ptr<float[]> nSize = nullptr;

public:
	Affine(ActivationName activationName, OptimizerName optName, ActivationName topActivationName, UINT inW, UINT inH, UINT* numNode,
		int depth, UINT split, UINT inputsetnum);
	void Draw(float x, float y);
	void Draw3D();
	void InConnection();
	void ErrConnection(bool update);
	void TestConnection();
	void DetectionConnection(UINT SearchNum);
};

class Pooling :public DxPooling {

protected:
	friend CNN;
	Affine* inAf = nullptr;
	Convolution* inCn = nullptr;
	Affine* errAf = nullptr;
	Convolution* errCn = nullptr;
	PolygonData2D dpo;
	UINT NumFilter;

public:
	Pooling(UINT width, UINT height, UINT poolNum, UINT inputsetnum);
	void Draw(float x, float y);
	void InConnection();
	void ErrConnection(bool update);
	void TestConnection();
	void DetectionConnection(UINT SearchNum, bool GradCAM_ON = false);
};

class Convolution :public DxConvolution {

protected:
	friend CNN;
	Affine* inAf = nullptr;
	Pooling* inPo = nullptr;
	Convolution* inCn = nullptr;
	Affine* errAf = nullptr;
	Convolution* errCn = nullptr;
	Pooling* errPo = nullptr;
	PolygonData2D dcn;
	UINT NumFilter;

public:
	Convolution(ActivationName activationName, OptimizerName optName, UINT width, UINT height, UINT filNum, bool DeConvolutionMode, UINT inputsetnum, UINT elnumwid, UINT filstep);
	void Draw(float x, float y);
	void InConnection();
	void ErrConnection(bool update);
	void TestConnection();
	void DetectionConnection(UINT SearchNum, bool GradCAM_ON = false);
};

enum LayerName {
	NONE,
	CONV,
	POOL,
	AFFINE
};

struct Layer {
	UINT maxThread;//�ő�X���b�h��(�S���C���[����)
	UINT mapWid;//���o�͈�wid
	UINT mapHei;//���o�͈�hei
	LayerName layerName = NONE;
	UINT NumFilter;
	UINT NumConvFilterWid;//��ݍ��݃t�B���^�[�T�C�Y
	UINT NumConvFilterSlide;//��ݍ��݃t�B���^�[�X���C�h��
	bool DeConvolutionMode = false;
	UINT numNode[MAX_DEPTH_NUM - 1];//Affine�̃m�[�h��(���͑w����)
	UINT topNodeWid = 1;//Affine�̎���Node�֏o�͂���ۂ�widSize
	UINT NumDepthNotInput;//Affine�̐[��(���͏���)
	ActivationName acName;//�������֐���
	OptimizerName optName = SGD;//�I�v�e�B�}�C�U�[
	ActivationName topAcName = CrossEntropySigmoid;//�������֐����ŏI�o��(Affine�̂�)
};

class CNN {

protected:
	UINT NumConv = 0;
	UINT NumPool = 0;
	Convolution** cn = nullptr;
	Pooling** po = nullptr;
	Affine* nn = nullptr;
	GradCAM* gc = nullptr;
	UINT layerSize = 0;
	LayerName firstLayer;
	LayerName endLayer;

	CNN() {}

public:
	CNN(UINT srcW, UINT srcH, Layer* layer, UINT layerSize);
	~CNN();
	void Detection(UINT SearchNum);
	void DetectionGradCAM(UINT SearchNum, UINT srcMapWid, UINT mapslide);
	void SetLearningLate(float nn, float cn, float cnB = 0.0f);
	void setOptimizerParameter(float LearningRateNN = 0.001f, float AttenuationRate1NN = 0.9f,
		float AttenuationRate2NN = 0.999f, float DivergencePreventionNN = 0.00000001f,
		float LearningRateCN = 0.001f, float AttenuationRate1CN = 0.9f,
		float AttenuationRate2CN = 0.999f, float DivergencePreventionCN = 0.00000001f,
		float LearningRateCNB = 0.001f, float AttenuationRate1CNB = 0.9f,
		float AttenuationRate2CNB = 0.999f, float DivergencePreventionCNB = 0.00000001f);
	void SetActivationAlpha(float nn, float cn);
	void SetdropThreshold(float* dropNN, float dropCN);
	void Training();
	void TrainingFp();
	void TrainingBp();
	void TrainingBpNoUpdate();
	float GetcrossEntropyError();
	float GetcrossEntropyErrorTest();
	void Test();
	void TrainingDraw(float x = 0.0f, float y = 0.0f);
	void GradCAMDraw(float x = 50.0f, float y = 200.0f);
	float GetOutputEl(UINT ElNum, UINT inputsetInd = 0);
	ID3D12Resource* GetOutputResource();
	ID3D12Resource* GetOutErrResource();
	void SetInErrResource(ID3D12Resource* res);
	void SetTargetEl(float el, UINT ElNum);
	void FirstInput(float el, UINT ElNum, UINT inputsetInd = 0);
	void InputArray(float* inArr, UINT arrNum, UINT inputsetInd = 0);
	void InputArrayEl(float el, UINT arrNum, UINT ElNum, UINT inputsetInd = 0);
	void SetPixel3ch(ID3D12Resource* pi);
	void SetPixel3ch(BYTE* pi);
	void SaveData();
	void LoadData();
};

#endif