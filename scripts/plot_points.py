import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import ndimage


#determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.

def get_isolated_points(points):
    isolated_inds=[]
    dist_mat = scipy.spatial.distance_matrix(points,points)
    avg=np.average(dist_mat)
    std=np.std(dist_mat)
    max_v=np.max(dist_mat)
    inds = np.where(dist_mat>(avg+30*std))
    print inds
    inds_x=inds[0]
    inds_y=inds[1]
    for i in range(len(inds_x)):
        if inds_x[i]<inds_y[i]:
            continue
        else:
            isolated_inds.append(inds_x[i])
    print isolated_inds
    return isolated_inds


def point_inside_polygon(x,y,poly):
    #http://www.ariel.com.au/a/python-point-int-poly.html
    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside



def make_mesh(points_x, points_y, stp):
    '''
    coord - outline coordinates
    stp - step size
    tolerance
    lng - 
    function res = model2mesh(coord,stp,tolerance,lng)
    % This function performs a medial axis transform to a non-branching axis
    % Takes the outline coordinates, step size on the final centerline,
    % tolerance to non-centrality, and the length of the ribs. The last
    % parameter should be longer than the ribs, but should not be too long to
    % intersect the countour agan: though most cases of >1 intersection will
    % be resolved by the function, some will not. Outputs the coordinates of
    % the centerline points.
    '''
    delta = 1E-10  
    #voronoi transform
    #First remove end points that are too close to the first point
    # while True:
    #     if len(points_x)<=2 :
    #          res=0
    #          return
    #     if abs(points_x[0]-points_x[-1])<0.00001 && abs(points_y[0]-[points_y[-1])<0.00001:
    #         coord = coord(1:end-1,:);
    #     else:
    #         break

    #run voronoi
    coords_xy = np.zeros([len(points_x),2])
    for i in range(len(points_x)):
        coords_xy[i,0]=points_x[i]
        coords_xy[i,1]=points_y[i]
    vor = scipy.spatial.Voronoi(coords_xy)
    print dir(vor)
    sys.exit()
    xx=[]
    yy=[]
    for ii in vor.vertices:
        if point_inside_polygon(ii[0],ii[1],coords_xy):
            xx.append(ii[0])
            yy.append(ii[1])

    coords_xy_in = np.zeros([len(xx),2])
    for ii in range(len(xx)):
        coords_xy_in[ii,0]=xx[ii]
        coords_xy_in[ii,1]=yy[ii]
    
    # inds = get_isolated_points(coords_xy_in)
    # xx_new=[]
    # yy_new=[]
    # for ii in range(len(coords_xy_in)):
    #     if ii not in inds:
    #         xx_new.append(xx[ii])
    #         yy_new.append(yy[ii])

    #remove branches
    # vx2=[]
    # vy2=[]
    # while True:
    #     for i in range(len(xx)):
    #         if (np.sum(np.sum((np.abs(vx-vx(0,i))<delta)&(abs(vy-vy(0,i))<delta)))>1)&&
    #            (np.sum(np.sum((np.abs(vx-vx(1,i))<delta)&(abs(vy-vy(1,i))<delta)))>1):
            
                        
    #             # vx2=[vx2 vx(:,i)];
    #             # vy2=[vy2 vy(:,i)];
    # #     if size(vx,2)-size(vx2,2)<=2,
    # #         vx3 = vx2;
    # #         vy3 = vy2;
    # #         break;
    # #     else
    # #         vx = vx2;
    # #         vy = vy2;
    # #         vx2 = [];
    # #         vy2 = [];
    # #     end
    # # end
    # # vx3 = vx;
    # # vy3 = vy;

            


            
    # fig, ax = plt.subplots()
    # ax.scatter(points_x,points_y,color='blue') ##    fig, ax = plt.subplots()
    # #ax.set_aspect('equal')
    # ax.scatter(xx_new,yy_new,color='red') ##    fig, ax = plt.subplots()

    # #plot all of the outline points
    
    # #plot all of the voroni points

    # #scipy.spatial.voronoi_plot_2d(vor)
    # plt.show()
    # print dir(vor)
    # sys.exit()
    '''
    
    % remove isolated points
    t = ~((abs(vx(1,:)-vx(2,:))<delta)&(abs(vy(1,:)-vy(2,:))<delta));
    vx = reshape(vx([t;t]),2,[]);
    vy = reshape(vy([t;t]),2,[]);
         
    % remove branches
    vx2=[];
    vy2=[];
    while true
        for i=1:size(vx,2)
            if((sum(sum((abs(vx-vx(1,i))<delta)&(abs(vy-vy(1,i))<delta)))>1)&&(sum(sum((abs(vx-vx(2,i))<delta)&(abs(vy-vy(2,i))<delta)))>1))
                vx2=[vx2 vx(:,i)];
                vy2=[vy2 vy(:,i)];
            end
        end
        if size(vx,2)-size(vx2,2)<=2,
            vx3 = vx2;
            vy3 = vy2;
            break;
        else
            vx = vx2;
            vy = vy2;
            vx2 = [];
            vy2 = [];
        end
    end
    vx3 = vx;
    vy3 = vy;
    
    % check that there are no cycles
    vx2 = [];
    vy2 = [];
    while size(vx,2)>1
        for i=1:size(vx,2)
            if((sum(sum((abs(vx-vx(1,i))<delta)&(abs(vy-vy(1,i))<delta)))>1)&&(sum(sum((abs(vx-vx(2,i))<delta)&(abs(vy-vy(2,i))<delta)))>1))
                vx2=[vx2 vx(:,i)];
                vy2=[vy2 vy(:,i)];
            end
        end
        if size(vx,2)-size(vx2,2)<=1
            res = 0;
            return;
        else
            vx = vx2;
            vy = vy2;
            vx2 = [];
            vy2 = [];
        end
    end
    vx = vx3;
    vy = vy3;
    if isempty(vx) || size(vx,1)<2, res=0;return;end

    % sort points
    vx2=[];
    vy2=[];
    for i=1:size(vx,2) % in this cycle find the first point
        if sum(sum(abs(vx-vx(1,i))<delta & abs(vy-vy(1,i))<delta))==1
            vx2=vx(:,i)';
            vy2=vy(:,i)';
            break
        elseif sum(sum(abs(vx-vx(2,i))<delta & abs(vy-vy(2,i))<delta))==1
            vx2=fliplr(vx(:,i)');
            vy2=fliplr(vy(:,i)');
            break
        end
    end
    k=2;
    while true % in this cycle sort all points after the first one
        f1=find(abs(vx(1,:)-vx2(k))<delta & abs(vy(1,:)-vy2(k))<delta & (abs(vx(2,:)-vx2(k-1))>=delta | abs(vy(2,:)-vy2(k-1))>=delta));
        f2=find(abs(vx(2,:)-vx2(k))<delta & abs(vy(2,:)-vy2(k))<delta & (abs(vx(1,:)-vx2(k-1))>=delta | abs(vy(1,:)-vy2(k-1))>=delta));
        if f1>0
            vx2 = [vx2 vx(2,f1)];
            vy2 = [vy2 vy(2,f1)];
        elseif f2>0
            vx2 = [vx2 vx(1,f2)];
            vy2 = [vy2 vy(1,f2)];
        else
            break
        end
        k=k+1;
    end

    skel=[vx2' vy2'];
    
    if size(vx2,2)<=1
        res = 0;
        return;
    end

    % interpolate skeleton to equal step, extend outside of the cell and smooth
    % tolerance=0.001;
    d=diff(skel,1,1);
    l=cumsum([0;sqrt((d.*d)*[1 ;1])]);
    if l(end)>=stp
        skel = [interp1(l,vx2,0:stp:l(end))' interp1(l,vy2,0:stp:l(end))'];
    else
        skel = [vx2(1) vy2(1);vx2(end) vy2(end)];
    end
    if size(skel,1)<=1 % || length(skel)<4
        res = 0;
        return;
    end
    lng0 = l(end);
    sz = lng0/stp;
    L = size(coord,1);
    coord = [coord;coord(1,:)];
    % lng = 100;
    skel2 = [skel(1,:)*(lng/stp+1) - skel(2,:)*lng/stp; skel;...
           skel(end,:)*(lng/stp+1) - skel(end-1,:)*lng/stp];
    d=diff(skel2,1,1);
    l=cumsum([0;sqrt((d.*d)*[1 ;1])]);
    [l,i] = unique(l);
    skel2 = skel2(i,:);
    if length(l)<2 || size(skel2,1)<2, res=0; return; end
    % find the intersection of the 1st skel2 segment with the contour, the
    % closest to one of the poles (which will be called 'prevpoint')
    [tmp1,tmp2,indS,indC]=intxyMulti(skel2(2:-1:1,1),skel2(2:-1:1,2),coord(:,1),coord(:,2));
    [tmp1,prevpoint] = min([min(modL(indC,1)) min(modL(indC,L/2+1))]);
    if prevpoint==2, prevpoint=L/2; end
    % prevpoint = mod(round(indC(1))-1,L)+1;
    skel3=spsmooth(l,skel2',tolerance,0:stp:l(end))';%1:stp:

    % recenter and smooth again the skeleton
    [pintx,pinty,q]=skel2mesh(skel3);
    if length(pintx)<sz-1
        skel3=spsmooth(l,skel2',tolerance/100,0:stp:l(end))';
        [pintx,pinty,q]=skel2mesh(skel3);
    end
    if ~q || length(pintx)<sz-1  || length(skel3)<4
        res=0;
        return
    end
    skel = [mean(pintx,2) mean(pinty,2)];
    d=diff(skel,1,1);
    l=cumsum([0;sqrt((d.*d)*[1 ;1])]);
    [l,i] = unique(l);
    skel = skel(i,:);
    if length(l)<2 || size(skel,1)<2, res=0; return; end
    skel=spsmooth(l,skel',tolerance,-3*stp:stp:l(end)+4*stp)';
    
    % get the mesh
    [pintx,pinty,q]=skel2mesh(skel);
    if ~q 
        res=0;
        return
    end
    res = [pintx(:,1) pinty(:,1) pintx(:,2) pinty(:,2)];
    if (pintx(1,1)-coord(end,1))^2+(pinty(1,1)-coord(end,2))^2 > (pintx(end,1)-coord(end,1))^2+(pinty(end,1)-coord(end,2))^2
        res = flipud(res);
    end
    if numel(res)==1 || length(res)<=4, res=0; gdisp('Unable to create mesh'); end
    if length(res)>1 && (res(1,1)~=res(1,3) || res(end,1)~=res(end,3)), res=0; gdisp('Mesh creation error! Cell rejected'); end

'''




    
    None


#load points
ff=open("/Users/kerenlasker/projects/opto/microscopy/scripts/data/cell_coord.txt")
lines = ff.readlines()
ff.close()
points_x=[]
points_y=[]
for line in lines:
    s=line.split()
    points_x.append(float(s[0]))
    points_y.append(float(s[1])) 

make_mesh(points_x, points_y, 1)
sys.exit()
    
fig, ax = plt.subplots()
ax.scatter(points_x,points_y) ##    fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.show()


'''''
Trying to understand how MicrobeTracker picks cells.
It it able to get the coutor before building the mesh.
The important function is getRegions in microbeTracker and there are few bwmorph calls that seems important.
The relevant module in python seems to be : http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.clear_border
http://scikit-image.org/docs/dev/api/skimage.morphology.html
'''''
