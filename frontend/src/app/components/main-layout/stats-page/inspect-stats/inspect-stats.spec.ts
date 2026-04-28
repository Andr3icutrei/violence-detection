import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InspectStats } from './inspect-stats';

describe('InspectStats', () => {
  let component: InspectStats;
  let fixture: ComponentFixture<InspectStats>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InspectStats],
    }).compileComponents();

    fixture = TestBed.createComponent(InspectStats);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
