const assert = require('node:assert/strict');

const { nextMicLevel, buildBarScales, buildOrbStyle } = require('../mic-level.js');


function testSpeechAttackAndSilenceDecay() {
  const speech = nextMicLevel(0, 0.08, 0.25);
  const quieter = nextMicLevel(speech, 0, 0);

  assert.ok(speech > 0.3, `expected visible speech level, received ${speech}`);
  assert.ok(quieter > 0, 'decay should remain smooth instead of snapping to zero');
  assert.ok(quieter < speech, 'silence should lower the displayed level');
}


function testBarScalesStayVisibleAndReactToVolume() {
  const quiet = buildBarScales(0, 0, 11);
  const speech = buildBarScales(1, Math.PI / 3, 11);

  assert.equal(speech.length, 11);
  assert.ok(quiet.every(value => value >= 0.1), 'idle bars should remain visible');
  assert.ok(speech.every(value => value >= 0.1 && value <= 1));
  assert.ok(Math.max(...speech) > Math.max(...quiet) + 0.5);
}


function testOrbStyleExpandsWithVolume() {
  const quiet = buildOrbStyle(0);
  const loud = buildOrbStyle(1);

  assert.equal(quiet.scale, 1);
  assert.equal(loud.scale, 1.42);
  assert.ok(loud.ringOpacity > quiet.ringOpacity);
  assert.ok(loud.coreShadow > quiet.coreShadow);
}


testSpeechAttackAndSilenceDecay();
testBarScalesStayVisibleAndReactToVolume();
testOrbStyleExpandsWithVolume();
console.log('mic-level tests passed');
