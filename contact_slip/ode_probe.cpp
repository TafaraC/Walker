// Offline ODE integration probe. No SimSpark or network is started.
#include <ode/ode.h>
int main() {
    dInitODE2(0);
    auto world = dWorldCreate();
    auto group = dJointGroupCreate(0);
    auto plane = dCreatePlane(0,0,0,1,0);
    auto wall = dCreatePlane(0,1,0,0,0);
    auto ball = dCreateSphere(0,.1);
    dContact c{};
    c.surface.mode = dContactSlip1 | dContactSlip2;
    c.surface.mu = 1;
    c.surface.slip1 = c.surface.slip2 = .005;
    c.geom.g1 = plane; c.geom.g2 = ball;
    c.geom.normal[2] = 1;
    for (int i=0; i<1000; ++i) dJointCreateContact(world,group,&c);
    c.geom.g1 = wall;
    dJointCreateContact(world,group,&c); // must not be changed or logged
    c.geom.g1 = plane; c.geom.pos[0] = 16;
    dJointCreateContact(world,group,&c); // out of field: unchanged
    dJointGroupDestroy(group);
    dGeomDestroy(ball); dGeomDestroy(wall); dGeomDestroy(plane);
    dWorldDestroy(world);
    dCloseODE();
}
