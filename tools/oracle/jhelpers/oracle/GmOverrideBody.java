package oracle;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.bodies.CelestialBody;
import org.orekit.frames.Frame;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.utils.TimeStampedFieldPVCoordinates;
import org.orekit.utils.TimeStampedPVCoordinates;

/** Delegating CelestialBody whose only change is getGM(): lets the oracle use gm_de431 constants
 *  while positions still come from Orekit's own JPL DE reader. */
public class GmOverrideBody implements CelestialBody {
    private final CelestialBody delegate;
    private final double gm;

    public GmOverrideBody(CelestialBody delegate, double gm) {
        this.delegate = delegate;
        this.gm = gm;
    }

    @Override public Frame getInertiallyOrientedFrame() { return delegate.getInertiallyOrientedFrame(); }
    @Override public Frame getBodyOrientedFrame() { return delegate.getBodyOrientedFrame(); }
    @Override public String getName() { return delegate.getName(); }
    @Override public double getGM() { return gm; }
    @Override public TimeStampedPVCoordinates getPVCoordinates(AbsoluteDate date, Frame frame) {
        return delegate.getPVCoordinates(date, frame);
    }
    @Override public Vector3D getPosition(AbsoluteDate date, Frame frame) { return delegate.getPosition(date, frame); }
    @Override public <T extends CalculusFieldElement<T>> FieldVector3D<T> getPosition(FieldAbsoluteDate<T> date, Frame frame) {
        return delegate.getPosition(date, frame);
    }
    @Override public <T extends CalculusFieldElement<T>> TimeStampedFieldPVCoordinates<T> getPVCoordinates(FieldAbsoluteDate<T> date, Frame frame) {
        return delegate.getPVCoordinates(date, frame);
    }
}
