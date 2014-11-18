import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import spatial
import shapely.geometry
import matplotlib.patches

def get_region(l1,l2,outline):
    #create a polygon from the two lines
    #extend l1
    d1=[l1[1][0]-l1[0][0],l1[1][1]-l1[0][1]]
    p1=[l1[0][0]-d1[0]*10,l1[0][1]-d1[1]*10]
    p2=[l1[1][0]+d1[0]*10,l1[1][1]+d1[1]*10]
    d2=[l2[1][0]-l2[0][0],l2[1][1]-l2[0][1]]
    p3=[l2[0][0]-d2[0]*10,l2[0][1]-d2[1]*10]
    p4=[l2[1][0]+d2[0]*10,l2[1][1]+d2[1]*10]
    points_order=[p1,p2,p3,p4]
    if shapely.geometry.Point(p2).distance(shapely.geometry.Point(p3))>shapely.geometry.Point(p2).distance(shapely.geometry.Point(p4)):
        points_order=[p1,p2,p4,p3]
    large_region=shapely.geometry.Polygon(points_order)
    g1 = shapely.geometry.Polygon(outline)
    region=g1.intersection(large_region)
    if not isinstance(region,shapely.geometry.polygon.Polygon):
        return None
    print type(region)
    rect = matplotlib.patches.Polygon(list(region.exterior.coords),color='g')
    #plt.add_patch(rect)
    return region
    
def create_mesh2(skel,outline,ax):
    #for each point on the skel find the closest point on the outline and draw a line
    #make a polygon out of the outline
    lines=[]
    poly = shapely.geometry.Polygon(outline)
    if not poly.is_valid:
        print "ERROR: The polygon is not valid. Exiting"
        l1 = shapely.geometry.LineString(poly.exterior.coords)
        if not l1.is_simple:
            print "ERROR: The polydon exterioir is not simple"
        return
    #find intersection points for each point of the outline
    print range(1,len(skel)-1),len(skel)
    dp=[skel[-1][1]-skel[0][1],-(skel[-1][0]-skel[0][0])]
    for i in range(0,len(skel)):
        try:
            p1=skel[i]
            line = shapely.geometry.LineString([p1,[p1[0]+dp[0]*200, p1[1]+dp[1]*200]])
            xx = poly.intersection(line)
            if isinstance(xx,shapely.geometry.MultiLineString):
                for linei in xx:
                    if shapely.geometry.Point(p1).distance(shapely.geometry.Point(linei.coords[0])) <0.01:
                        ip1=linei.coords[1]
                    else:
                        continue 
            else:
                ip1=xx.coords[1]
      
            line = shapely.geometry.LineString([p1,[p1[0]-dp[0]*20, p1[1]-dp[1]*20]])
            xx = poly.intersection(line)
            if isinstance(xx,shapely.geometry.MultiLineString):
                for linei in xx:
                    if shapely.geometry.Point(p1).distance(shapely.geometry.Point(linei.coords[0])) <0.01:
                        ip2=linei.coords[1]
                    else:
                        continue 
            else:
                ip2=xx.coords[1]
            lines.append([ip1,ip2])
        except:
            print "did not draw a line for",i
    print lines
    #draw the mesh
    for i in range(len(lines)-1):
        region = get_region(lines[i],lines[i+1],outline)
        if region==None:
            continue
        print list(region.exterior.coords)
        rect = matplotlib.patches.Polygon(list(region.exterior.coords),color=np.random.rand(3,1))
        ax.add_patch(rect)
   

    
    return lines


def modL(a,shift):
    L=80
    out = np.mod(a-shift,L)
    out = np.minimum(out,L-out)
    return out


def intxyMulti(ax,ay,bx,by,jstart=1,jdir=1):
    '''
    Finds all intersection points between lines ax,ay and bx,by
    % Outputs intersection coordinates aintx,ainty and the position on a and b,
    % counted from 1 (first point) to n (last point), real number. All outputs
    % are vectors if multiple intersections occurs
    % If any other argument provided, only the first intersection point is
    % given
    % If 2 additional arguments provided, the first one is the starting point,
    % the next one (1 or -1) the direction (in the second/"b" variable)

    if isempty(varargin), jstart = -1; jdir = 1;
    elseif length(varargin)==1, jstart = 1; jdir = 1;
    elseif length(varargin)==2, jstart = varargin{1}; jdir = varargin{2};
    end
    [aintx,ainty,aia,aib]=intxyMultiC(ax,ay,bx,by,jstart,jdir);
    '''

    multiout = True
    print "range",jstart,len(bx)*jdir+jstart-1,len(bx),jdir
    end_num=len(bx)*jdir+jstart-1
    num_steps = np.abs(end_num-jstart)+1
    jrange = np.mod(np.linspace(jstart,end_num,num=num_steps),len(bx))
    print jrange
    aintx = [];
    ainty = [];
    aia = [];
    aib = [];
    print "len ax",len(ax),ax
    for i in range(len(ax)-1):
        #vector to next vertex in a
        u=np.array([ax[i+1]-ax[i],ay[i+1]-ay[i]]);
        #normalize u
        magu=math.sqrt(u[0]*u[0]+u[1]*u[1])
        if magu==0:
            print "magu is zero"
            continue
        u=u/magu; 
        #go through each vertex in b
        for j in jrange:
            #check for intersection
            if ax[i]==bx[j] and ay[i]==by[j]:
                aintx = np.append(aintx,ax[i])
                ainty = np.append(ainty,ay[i])
                aia = np.append(aia,i+1)##
                aib = np.append(aib,j+1)##
                continue
            #vector from ith vertex in a to jth vertex in b
            v=np.array([bx[j]-ax[i],by[j]-ay[i]])
            #vector from ith+1 vertex in a to jth vertex in b
            w=np.array([bx[j]-ax[i+1],by[j]-ay[i+1]])
            #check whether a and b crossed
            if j>0:
                #vector from ith vertex of a to jth-1 vertex of b
                vv=[bx[j-1]-ax[i], by[j-1]-ay[i]];
                #vector from jth-1 vertex of b to jth vertex of b
                z=np.array([bx[j]-bx[j-1], by[j]-by[j-1]])
                #cross product of u and v
                cpuv=u[0]*v[1]-u[1]*v[0];
                #cross product of u and vv
                cpuvv=u[0]*vv[1]-u[1]*vv[0];
                #cross product of v and z
                cpvz=v[0]*z[1]-v[1]*z[0];
                #cross product of w and z
                cpwz=w[0]*z[1]-w[1]*z[0];   
                if (cpuv*cpuvv<=0) and (cpvz*cpwz<0):
                    #normalize v
                    magv=math.sqrt(v[0]*v[0]+v[1]*v[1]);
                    if magv==0:
                         continue
                    v=v/magv;
                    #normalize z
                    magz=math.sqrt(z[0]*z[0]+z[1]*z[1])
                    if magz==0:
                        continue
                    z=z/magz
                    #ith segment of a crosses jth segment of b
                    #range from a(i) to intersection (Law of Sines)
                    r=magv*math.sin(math.acos(np.dot(v,z)))/math.sin(math.acos(np.dot(-u,z)));
                    if (cpuv*cpuvv<=0) and (r==1):
                         continue
                    #record index along a to include portion between ith and
                    #ith+1 vertex where cpa occurs
                    ia=i+r/magu;
                    #project u along x and y to find point of itersection
                    intx=ax[i]+r/magu*(ax[i+1]-ax[i]);
                    inty=ay[i]+r/magu*(ay[i+1]-ay[i]);
                    #range from b(j) to intersection 
                    r=math.sqrt((bx[j]-intx)*(bx[j]-intx)+(by[j]-inty)*(by[j]-inty));
                    if (cpuv*cpuvv<=0) and (r==1):
                        continue
                    #record index along b to include portion between jth-1 and
                    #jth vertex where intersection occurs
                    ib=j-r/magz;
                    #assign to output variables
                    aintx = np.append(aintx,intx)
                    ainty = np.append(ainty,inty)
                    aia = np.append(aia,ia+1)##
                    aib =np.append(aib,ib+1)##
                    #if ~multiout:
                    #    return
    print "going to return"
    return [aintx,ainty,aia,aib]


def skeleton_to_mesh(sk,coord,prevpoint,lng,stp):
    print "h1"
    pintx=[];
    pinty=[];
    q=False
    if len(sk)==0:
        return [pintx,pinty,q]
    print "h2",len(sk[:,0]),len(sk[:,1]),len(coord[:,0]),len(coord[:,1])
    #Find the intersection of the skel with the contour closest to prevpoint
    [intX,intY,indS,indC]=intxyMulti(sk[:,0],sk[:,1],coord[:,0],coord[:,1]);
    print "h3"
    if len(intX)==0 or len(indC)==0 : 
        print "h4",len(intX),len(indC)
        return [pintx,pinty,q]
    ind = (modL(indC,prevpoint)).argmin()
    prevpoint = indC[ind]
    print ind,prevpoint
    indS=indS[ind]
    if indS>(len(sk)-indS):
        sk = sk[:math.ceil(indS),:][::-1,:]
    else:
        sk = sk[math.floor(indS)-1:,:]
    #2. define the first pair of intersections as this point
    # 3. get the list of intersections for the next pair
    # 4. if more than one, take the next in the correct direction
    # 5. if no intersections found in the reqion between points, stop
    # 6. goto 3.
    # Define the lines used to compute intersections
    d=np.zeros([len(sk)-1,2])
    for i in range(1,len(sk)):
        d[i-1,0]=sk[i,0]-sk[i-1,0]
        d[i-1,1]=sk[i,1]-sk[i-1,1]
    #make plinesx1
    plinesx11= np.matlib.repmat(sk[:-1,0],2,1)
    plinesx1 = np.zeros([len(plinesx11[0]),2])
    plinesx1[:,0]=plinesx11[0]
    plinesx1[:,1]=plinesx11[0] +lng/stp*d[:,1]
    #make plinesy1
    plinesy11= np.matlib.repmat(sk[:-1,1],2,1)
    plinesy1 = np.zeros([len(plinesy11[0]),2])
    plinesy1[:,0]=plinesy11[0]
    plinesy1[:,1]=plinesy11[0] -lng/stp*d[:,0]
    #make plinesx2
    plinesx21= np.matlib.repmat(sk[:-1,0],2,1)
    plinesx2 = np.zeros([len(plinesx21[0]),2])
    plinesx2[:,0]=plinesx21[0]
    plinesx2[:,1]=plinesx21[0] -lng/stp*d[:,1]
    #make plinesy2
    plinesy21= np.matlib.repmat(sk[:-1,1],2,1)
    plinesy2 = np.zeros([len(plinesy21[0]),2])
    plinesy2[:,0]=plinesy21[0]
    plinesy2[:,1]=plinesy21[0] +lng/stp*d[:,0]

    # Allocate memory for the intersection points
    pintx = np.zeros([len(sk)-1,2])
    pinty = np.zeros([len(sk)-1,2])
    # Define the first pair of intersections as the prevpoint
    pintx[0,:] = [intX[ind],intX[ind]];
    pinty[0,:] = [intY[ind],intY[ind]];
    prevpoint1 = prevpoint;
    prevpoint2 = prevpoint;
    #for i=1:size(d,1), plot(plinesx(i,:),plinesy(i,:),'r'); end % Testing
    q=True;
    fg = 1;
    jmax = len(sk)-1;
    for j in range(jmax):
        [pintx1,pinty1,tmp,indC1]=intxyMulti(plinesx1[j,:],plinesy1[j,:],coord[:,0],coord[:,1],math.floor(prevpoint1),1);
        [pintx2,pinty2,tmp,indC2]=intxyMulti(plinesx2[j,:],plinesy2[j,:],coord[:,0],coord[:,1],math.ceil(prevpoint2),-1);
        if (j==0):
            print [pintx1,pinty1,tmp,indC1]
        if (len(pintx1)>0) and (len(pintx2)>0):
            if pintx1!=pintx2:
                if fg==3:
                    break
                fg = 2
                ind1 = (modL(indC1,prevpoint1)).argmin()
                ind2 = (modL(indC2,prevpoint2)).argmin()
                prevpoint1 = indC1[ind1]
                prevpoint2 = indC2[ind2]
                pintx[j,:]=[pintx1[ind1],pintx2[ind2]];
                pinty[j,:]=[pinty1[ind1],pinty2[ind2]];
            else:
                q=False;
                return [pintx,pinty,q]
        elif fg==2:
            fg = 3
    inds=np.where(pintx[:,0]!=0)
    pinty = pinty[inds,:]
    pintx = pintx[inds,:]
    [intX,intY,indS,indC]=intxyMulti(sk[:,0],sk[:,1],coord[:,0],coord[:,1]);
    ind = (modL(indC,prevpoint)).argmax()
    prevpoint=(modL(indC,prevpoint))[ind]
    print pintx[0]
    pintx = np.append(pintx[0],np.array([intX[ind],intX[ind]]))
    pinty = np.append(pinty,np.array([intY[ind],intY[ind]]))
    print "===|||||||===",pintx
    sys.exit()
    pintx= pintx.reshape(len(pintx)/2,2)
    pinty= pinty.reshape(len(pinty)/2,2)
    nonan = (~np.isnan(pintx[:,0]))&(~np.isnan(pinty[:,0]))&(~np.isnan(pintx[:,1]))&(~np.isnan(pinty[:,1]))
    pintx = pintx[nonan,:]
    pinty = pinty[nonan,:]
    return [pintx,pinty,q]
    
if __name__=="__main__":

    #read the skeleton coordinates
    ff=open('/Users/kerenlasker/projects/opto/microscopy/scripts/data/skeleton_coords.txt')
    #ff=open('/Users/kerenlasker/projects/opto/microscopy/scripts/data/cell_outline2.txt')
    lines = ff.readlines()
    ff.close()
    sk_x=[]
    sk_y=[]
    for line in lines:
        s=line.split()
        sk_x.append(float(s[0]))
        sk_y.append(float(s[1])) 
    skel = np.zeros([len(sk_x),2])
    n_points=len(sk_x)
    for i in range(len(sk_x)):
        skel[i,0]=sk_x[i]
        skel[i,1]=sk_y[i]

    #read the outline coordinates
    ff=open("/Users/kerenlasker/projects/opto/microscopy/scripts/data/cell_outline.txt")
    lines = ff.readlines()
    ff.close()
    points_x=[]
    points_y=[]
    for line in lines:
        s=line.split()
        points_x.append(float(s[0]))
        points_y.append(float(s[1])) 
    outline = np.zeros([len(points_x),2])
    for i in range(len(points_x)):
        outline[i,0]=points_x[i]
        outline[i,1]=points_y[i]

    prevpoint=1
    lng=15
    stp=1
    [pintx,pinty,q]=skeleton_to_mesh(skel,outline,prevpoint,lng,stp)
    print "pintx"
    print pintx
    print "pinty"
    print pinty
    fig, ax = plt.subplots()
    
    ax.plot(np.transpose(pintx), np.transpose(pinty), 'k-')
    plt.show()



