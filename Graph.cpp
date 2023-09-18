//*****************************************************************************************//
//**                                                                                     **//
//**                                    Graph                                            **//
//**                                                                                     **//
//*****************************************************************************************//

#include "Graph.h"

Graph::Graph() {
	graph = new PolygonData2D();
	graph->SetName("graph_Graph");
	graph->GetVBarray2D(1);
}

Graph::~Graph() {
	S_DELETE(graph);
	ARR_DELETE(point);
}

void Graph::CreateGraph(float x, float y, float w, float h, int pw, int ph) {
	piw = pw;
	pih = ph;
	graph->TexOn();
	graph->TextureInit(pw, ph);
	graph->CreateBox(0, x, y, 0.1f, w, h, 1.0f, 1.0f, 1.0f, 1.0f, TRUE, TRUE);
	point = new UCHAR[ph * pw * 4];
	Clear();
}

void Graph::Clear() {
	for (int j = 0; j < pih * piw * 4; j++) {
		point[j] = 0;
	}
}

void Graph::SetData(int cnt, int data, UINT col) {
	point[data * piw * 4 + cnt * 4 + 0] = (col >> 24) & 0xff;
	point[data * piw * 4 + cnt * 4 + 1] = (col >> 16) & 0xff;
	point[data * piw * 4 + cnt * 4 + 2] = (col >> 8) & 0xff;
	point[data * piw * 4 + cnt * 4 + 3] = (col >> 0) & 0xff;
}

void Graph::Draw(int com_no) {
	graph->Update(0, 0, 0, 0, 0, 0, 0, 1.0f, 1.0f);
	graph->SetTextureMPixel(com_no, point, 0);
	graph->Draw(0);
}