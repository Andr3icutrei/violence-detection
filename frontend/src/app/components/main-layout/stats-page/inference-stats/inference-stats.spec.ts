import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InferenceStats } from './inference-stats';

describe('InferenceStats', () => {
  let component: InferenceStats;
  let fixture: ComponentFixture<InferenceStats>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InferenceStats],
    }).compileComponents();

    fixture = TestBed.createComponent(InferenceStats);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
