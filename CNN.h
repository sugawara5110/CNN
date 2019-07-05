//*****************************************************************************************//
//**                                                                                     **//
//**                                    CNN                                              **//
//**                                                                                     **//
//*****************************************************************************************//
#ifndef Class_CNN_Header
#define Class_CNN_Header

#include "../Common/Direct3DWrapper/Dx_NN.h"

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

public:
	Affine(ActivationName activationName, ActivationName topActivationName, UINT inW, UINT inH, UINT* numNode,
		int depth, UINT split, UINT inputsetnum);
	void Draw(float x, float y);
	void InConnection();
	void ErrConnection();
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
	void ErrConnection();
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
	Convolution(ActivationName activationName, UINT width, UINT height, UINT filNum, UINT inputsetnum, UINT elnumwid, UINT filstep);
	void Draw(float x, float y);
	void InConnection();
	void ErrConnection();
	void TestConnection();
	void DetectionConnection(UINT SearchNum, bool GradCAM_ON = false);
};

enum LayerName {
	CONV,
	POOL,
	AFFINE
};

struct Layer {
	UINT maxThread;//�ő�X���b�h��(�S���C���[����)
	UINT mapWid;//���o�͈�wid
	UINT mapHei;//���o�͈�hei
	LayerName layerName;
	UINT NumFilter;
	UINT NumConvFilterWid;//��ݍ��݃t�B���^�[�T�C�Y
	UINT NumConvFilterSlide;//��ݍ��݃t�B���^�[�X���C�h��
	UINT numNode[MAX_DEPTH_NUM - 1];//Affine�̃m�[�h��(���͑w����)
	UINT NumDepthNotInput;//Affine�̐[��(���͏���)
	ActivationName acName;//�������֐���
	ActivationName topAcName;//�������֐����ŏI�o��(Affine�̂�)
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
	void SetLearningLate(float nn, float cn);
	void Training();
	void TrainingFp();
	void TrainingBp();
	float GetcrossEntropyError();
	float GetcrossEntropyErrorTest();
	void Test();
	void TrainingDraw();
	void GradCAMDraw();
	void GetOutput(float* out, UINT inputsetInd = 0);
	float GetOutputEl(UINT ElNum, UINT inputsetInd = 0);
	ID3D12Resource* GetOutputResource();
	void SetTargetEl(float el, UINT ElNum);
	void FirstInput(float el, UINT ElNum, UINT inputsetInd = 0);
	void SetPixel3ch(ID3D12Resource* pi);
	void SetPixel3ch(BYTE* pi);
	void SaveData();
	void LoadData();
};

#endif