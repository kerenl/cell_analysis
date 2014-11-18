import skeleton_to_mesh
import utils2
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # ffx=open('/Users/kerenlasker/projects/opto/microscopy/scripts/vx.csv')
    # lines = ffx.readlines()
    # ffx.close()
    # edges_x1=np.zeros([len(lines),2])
    # for i,line in enumerate(lines):
    #     s=line.split(",")
    #     edges_x1[i,:]=[float(s[0]),float(s[1])]
    # ffy=open('/Users/kerenlasker/projects/opto/microscopy/scripts/vy.csv')
    # lines = ffy.readlines()
    # ffy.close()
    # edges_y1=np.zeros([len(lines),2])
    # for i,line in enumerate(lines):
    #     s=line.split(",")
    #     edges_y1[i,:]=[float(s[0]),float(s[1])]
    # [edges_x3,edges_y3]=utils.remove_branches(edges_x1,edges_y1)
    # print len(edges_x3)
    # print edges_x3
    # sys.exit()

    
    ff=open('/Users/kerenlasker/projects/opto/microscopy/scripts/data/cell_outline.txt')
    lines = ff.readlines()
    ff.close()
    ox=[]
    oy=[]
    for line in lines:
        s=line.split()
        ox.append(float(s[0]))
        oy.append(float(s[1]))
    outline=np.zeros([len(ox),2])
    outline[:,0]=ox
    outline[:,1]=oy
    skel = utils2.get_skeleton_from_outline(outline,ax)

    ax.plot(skel[:,0],skel[:,1],'ro')
    skeleton_to_mesh.create_mesh2(skel,outline,ax)
    plt.show()
    sys.exit()

    
    #find intersection
    import shapely.geometry

    poly = shapely.geometry.Polygon(outline)
    if not poly.is_valid:
        print "ERROR: The polygon is not valid. Exiting"
        l1 = shapely.geometry.LineString(poly.exterior.coords)
        if not l1.is_simple:
            print "ERROR: The polydon exterioir is not simple"
        sys.exit()
    
    p1=skel[0]
    p2=skel[1]
    d1=p1-p2
    dp=np.array([-d1[1],d1[0]])

    line = shapely.geometry.LineString([p1, p1+dp*100])
    xx = poly.intersection(line)
    print "line",line
    print "xx",xx
    print "intersection",line.intersects(poly)
    intersecting_points=[]
    for c in xx.coords:
        if shapely.geometry.Point(p1).distance(shapely.geometry.Point(c)) <0.01:
            continue
        else:
            intersecting_points.append(shapely.geometry.Point(c))
    print "Found",len(intersecting_points)
    for p in intersecting_points:
        print p



    line = shapely.geometry.LineString([p1, p1+dp-100])
    xx = poly.intersection(line)
    intersecting_points=[]
    for c in xx.coords:
        if shapely.geometry.Point(p1).distance(shapely.geometry.Point(c)) <0.01:
            continue
        else:
            intersecting_points.append(shapely.geometry.Point(c))
    print "Found other side",len(intersecting_points)
    for p in intersecting_points:
        print p

        


        
    sys.exit()
    #print 'Green line intersects clipped shape:', line2.intersects(clipped_shape)

'''    
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(*np.array(line).T, color='blue', linewidth=3, solid_capstyle='round')
ax.plot(*np.array(line2).T, color='green', linewidth=3, solid_capstyle='round')
ax.add_patch(descartes.PolygonPatch(clipped_shape, fc='blue', alpha=0.5))
ax.axis('equal')

plt.show()
    
    skel2=skel
    coord=outline
    lng=15
    stp=1
    L=80
    # [tmp1,tmp2,indS,indC]=skeleton_to_mesh.intxyMulti(skel2[2:,0],skel2[2:,1],coord[:,0],coord[:,1])
    # aa=np.array([min(skeleton_to_mesh.modL(indC,1)),min(skeleton_to_mesh.modL(indC,L/2+1))])
    # tmp1=aa.argmin()
    # prevpoint=aa[tmp1]
    # if prevpoint==1:
    #     prevpoint=L/2
    fig, ax = plt.subplots()
    prevpoint=L/2


    plt.scatter(outline[:,0],outline[:,1],c='g')
    for i in range(len(skel)):
        v0=skel[i]
        v1=skel[i]
        plt.plot([v0[0],v1[0]],[v0[1],v1[1]],'ro')
    plt.show()

    
    ax.plot(mesh_points_x,mesh_points_y)

    utils2.adjust_bounds(ax, outline)
    plt.show()
    sys.exit()
    
    print "before create mesh"
    skeleton_to_mesh.create_mesh(skel,outline)
    sys.exit()

    
    [pintx,pinty,q]=skeleton_to_mesh.skeleton_to_mesh(skel,outline,1,lng,stp)
    print "==========",len(pintx)
    mesh_points_x=[]
    mesh_points_y=[]
    for i in range(len(pintx)):
        mesh_points_x.append([pintx[i,0],pinty[i,0]])
        mesh_points_y.append([pintx[i,1],pinty[i,1]])
    ax.plot(mesh_points_x,mesh_points_y)

    utils2.adjust_bounds(ax, outline)
    plt.show()


   

    # a1=[12,11,1,1,2,3,4,5,6,9,6,7]
    # a2=[11,1,3,2,3,4,5,6,9,10,7,8]
    # utils2.remove_branches(a1,a2)
'''
