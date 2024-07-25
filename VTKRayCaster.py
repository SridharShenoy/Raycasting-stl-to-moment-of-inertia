import vtk


class rayCaster:

    def __init__(self, filenameSTL):
        readerSTL = vtk.vtkSTLReader()
        readerSTL.SetFileName(filenameSTL)
        readerSTL.Update()

        polydata = readerSTL.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            raise ValueError()
        self.mesh = polydata
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        self.bounds = mapper.GetBounds()
        self.obbTree = vtk.vtkOBBTree()
        self.obbTree.SetDataSet(self.mesh)
        self.obbTree.BuildLocator()

    def getBounds(self):
        return self.bounds

    def inside(self, src, tgt):
        pointsVTKintersection = vtk.vtkPoints()

        self.obbTree.IntersectWithLine(src, tgt, pointsVTKintersection, None)

        pointsVTKIntersectionData = pointsVTKintersection.GetData()
        noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
        # pointsIntersection = []
        # for idx in range(noPointsVTKIntersection):
        #    _tup = pointsVTKIntersectionData.GetTuple3(idx)
        #    pointsIntersection.append(_tup)
        return noPointsVTKIntersection

