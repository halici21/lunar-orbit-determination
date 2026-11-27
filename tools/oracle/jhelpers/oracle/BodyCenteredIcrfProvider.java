package oracle;

import org.hipparchus.CalculusFieldElement;
import org.orekit.bodies.CelestialBody;
import org.orekit.frames.FieldTransform;
import org.orekit.frames.Frame;
import org.orekit.frames.Transform;
import org.orekit.frames.TransformProvider;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;

/** Pure translation from an ICRF-aligned parent (GCRF) to a body-centred frame with ICRF axes.
 *  Orekit's CelestialBody.getInertiallyOrientedFrame() follows the IAU pole (it rotates at ~4e-9 rad/s
 *  for the Moon), so it cannot be used as a propagation frame for a non-rotating Moon-centred state. */
public class BodyCenteredIcrfProvider implements TransformProvider {
    private final CelestialBody body;
    private final Frame parent;

    public BodyCenteredIcrfProvider(CelestialBody body, Frame parent) {
        this.body = body;
        this.parent = parent;
    }

    @Override
    public Transform getTransform(AbsoluteDate date) {
        // coordinates of the parent origin in the child frame = -(body PV in parent)
        return new Transform(date, body.getPVCoordinates(date, parent).negate());
    }

    @Override
    public <T extends CalculusFieldElement<T>> FieldTransform<T> getTransform(FieldAbsoluteDate<T> date) {
        return new FieldTransform<>(date, body.getPVCoordinates(date, parent).negate());
    }
}
