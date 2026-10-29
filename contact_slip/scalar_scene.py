"""Versioned single-argument scene setter experiment; no installed-file edits."""
from configure_field_slip import SLIP_PATTERN
from variable_training.common import digest, save


def convert(directory, manifest, slip):
    """Convert only the generated scene and refresh its pre-import verification."""
    scene = directory/'soccer.rsg'
    text = scene.read_text()
    matches = list(SLIP_PATTERN.finditer(text))
    if len(matches) != 1 or matches[0].group('second') is None:
        raise ValueError('Expected one two-argument field setter before conversion')
    match = matches[0]
    if any(float(match.group(key)) != slip for key in ('first','second')):
        raise ValueError('Generated field setter differs from requested slip')
    old_hash = manifest['sha256']
    if digest(scene) != old_hash:
        raise ValueError('Generated scene changed before scalar conversion')
    replacement = f'{match.group("prefix")}{slip:.12g}{match.group("suffix")}'
    scene.write_text(text[:match.start()] + replacement + text[match.end():])
    new_hash = digest(scene)
    script = directory/'naosoccersim.rb'
    ruby = script.read_text()
    if ruby.count(old_hash) != 2:
        raise ValueError('Expected both import checksum and startup marker checksum')
    script.write_text(ruby.replace(old_hash,new_hash))
    result = dict(manifest)
    result.update(sha256=new_hash, startup_marker=manifest['startup_marker'].replace(old_hash,new_hash),
                  scene_setter_method='single_argument_v1',
                  field_contact_slip=slip, expected_field_contact_slip=slip,
                  matches_expected=True, legacy_generated_scene_sha256=old_hash,
                  config_sha256={p.name:digest(p) for p in directory.glob('*.rb')})
    save(directory/'environment_manifest.json',result)
    return result
